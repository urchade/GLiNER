import torch
import torch.nn.functional as F
from torch import nn

def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )


class SpanQuery(nn.Module):

    def __init__(self, hidden_size, max_width, trainable=True):
        super().__init__()

        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))

        nn.init.uniform_(self.query_seg, a=-1, b=1)

        if not trainable:
            self.query_seg.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        span_rep = torch.einsum('bld, ds->blsd', h, self.query_seg)

        return self.project(span_rep)


class SpanMLP(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.mlp = nn.Linear(hidden_size, hidden_size * max_width)

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.mlp(h)

        span_rep = span_rep.view(B, L, -1, D)

        return span_rep.relu()


class SpanCAT(nn.Module):

    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.query_seg = nn.Parameter(torch.randn(128, max_width))

        self.project = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),
            nn.ReLU()
        )

    def forward(self, h, *args):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        h = h.view(B, L, 1, D).repeat(1, 1, self.max_width, 1)

        q = self.query_seg.view(1, 1, self.max_width, -1).repeat(B, L, 1, 1)

        span_rep = torch.cat([h, q], dim=-1)

        span_rep = self.project(span_rep)

        return span_rep


class SpanConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, span_mode='conv_normal'):
        super().__init__()

        if span_mode == 'conv_conv':
            self.conv = nn.Conv1d(hidden_size, hidden_size,
                                  kernel_size=kernel_size)

            # initialize the weights
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')

        elif span_mode == 'conv_max':
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode == 'conv_mean' or span_mode == 'conv_sum':
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        self.span_mode = span_mode

        self.pad = kernel_size - 1

    def forward(self, x):

        x = torch.einsum('bld->bdl', x)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)

        x = self.conv(x)

        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)

        return torch.einsum('bdl->bld', x)


class SpanConv(nn.Module):
    def __init__(self, hidden_size, max_width, span_mode):
        super().__init__()

        kernels = [i + 2 for i in range(max_width - 1)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanConvBlock(hidden_size, kernel, span_mode))

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, *args):

        span_reps = [x]

        for conv in self.convs:
            h = conv(x)
            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-2)

        return self.project(span_reps)


class SpanEndpointsBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        B, L, D = x.size()

        span_idx = torch.LongTensor(
            [[i, i + self.kernel_size - 1] for i in range(L)]).to(x.device)

        x = F.pad(x, (0, 0, 0, self.kernel_size - 1), "constant", 0)

        # endrep
        start_end_rep = torch.index_select(x, dim=1, index=span_idx.view(-1))

        start_end_rep = start_end_rep.view(B, L, 2, D)

        return start_end_rep


class ConvShare(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(
            torch.randn(hidden_size, hidden_size, max_width))

        nn.init.kaiming_uniform_(self.conv_weigth, nonlinearity='relu')

        self.project = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, *args):
        span_reps = []

        x = torch.einsum('bld->bdl', x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, :i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        out = torch.stack(span_reps, dim=-2)

        return self.project(out)


def extract_elements(sequence, indices):
    B, L, D = sequence.shape
    K = indices.shape[1]

    # Expand indices to [B, K, D]
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)

    # Gather the elements
    extracted_elements = torch.gather(sequence, 1, expanded_indices)

    return extracted_elements


class SpanMarker(nn.Module):

    def __init__(self, hidden_size, max_width, dropout=0.4):
        super().__init__()

        self.max_width = max_width

        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
        )

        self.out_project = nn.Linear(hidden_size * 2, hidden_size, bias=True)

    def forward(self, h, span_idx):
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        # project start and end
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        # concat start and end
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        # project
        cat = self.out_project(cat)

        # reshape
        return cat.view(B, L, self.max_width, D)


class SpanMarkerV0(nn.Module):
    """
    Marks and projects span endpoints using an MLP.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat).view(B, L, self.max_width, D)


class ConvShareV2(nn.Module):
    def __init__(self, hidden_size, max_width):
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(
            torch.randn(hidden_size, hidden_size, max_width)
        )

        nn.init.xavier_normal_(self.conv_weigth)

    def forward(self, x, *args):
        span_reps = []

        x = torch.einsum('bld->bdl', x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, :i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        out = torch.stack(span_reps, dim=-2)

        return out


class SpanRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, span_mode, **kwargs):
        super().__init__()

        if span_mode == 'marker':
            self.span_rep_layer = SpanMarker(hidden_size, max_width, **kwargs)
        elif span_mode == 'markerV0':
            self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)
        elif span_mode == 'query':
            self.span_rep_layer = SpanQuery(
                hidden_size, max_width, trainable=True)
        elif span_mode == 'mlp':
            self.span_rep_layer = SpanMLP(hidden_size, max_width)
        elif span_mode == 'cat':
            self.span_rep_layer = SpanCAT(hidden_size, max_width)
        elif span_mode == 'conv_conv':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_conv')
        elif span_mode == 'conv_max':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_max')
        elif span_mode == 'conv_mean':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_mean')
        elif span_mode == 'conv_sum':
            self.span_rep_layer = SpanConv(
                hidden_size, max_width, span_mode='conv_sum')
        elif span_mode == 'conv_share':
            self.span_rep_layer = ConvShare(hidden_size, max_width)
        else:
            raise ValueError(f'Unknown span mode {span_mode}')

    def forward(self, x, *args):

        return self.span_rep_layer(x, *args)
