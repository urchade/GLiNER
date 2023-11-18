import torch
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch import nn

HIDDEN_SIZE_MULTIPLIER = 4


def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * HIDDEN_SIZE_MULTIPLIER),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * HIDDEN_SIZE_MULTIPLIER, out_dim)
    )


class SpanQuery(nn.Module):
    """
    Computes span representations using query vectors.
    """

    def __init__(self, hidden_size: int, max_width: int, trainable: bool = True, dropout: float = 0.4):
        super().__init__()
        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))
        nn.init.uniform_(self.query_seg, a=-1, b=1)
        self.query_seg.requires_grad = trainable
        self.project = create_projection_layer(hidden_size, dropout)

    def forward(self, input_tensor: torch.Tensor, *args) -> torch.Tensor:
        span_rep = torch.einsum('bld, ds->blsd', input_tensor, self.query_seg)
        return self.project(span_rep)


class SpanMLP(nn.Module):
    """
    Applies a Multi-Layer Perceptron (MLP) to each element of a sequence.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.mlp = create_projection_layer(hidden_size, dropout)
        self.out_dim = hidden_size * max_width

    def forward(self, input_tensor: torch.Tensor, *args) -> torch.Tensor:
        B, L, D = input_tensor.size()
        span_rep = self.mlp(input_tensor).view(B, L, -1, D).relu()
        return span_rep


class SpanEndpoints(nn.Module):
    """
    Applies a multi-layer projection to span endpoints and width embeddings.
    """

    def __init__(self, hidden_size: int, max_width: int, width_embedding: int = 128, dropout: float = 0.4):
        super().__init__()
        self.span_extractor = EndpointSpanExtractor(
            hidden_size, combination='x,y',
            num_width_embeddings=max_width, span_width_embedding_dim=width_embedding
        )
        self.downproject = create_projection_layer(hidden_size * 2 + width_embedding, dropout, hidden_size)

    def forward(self, input_tensor: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        span_rep = self.span_extractor(input_tensor, span_idx)
        return self.downproject(span_rep).view(input_tensor.size(0), input_tensor.size(1), -1, input_tensor.size(2))


class SpanConvBlock(nn.Module):
    """
    Applies either 1D convolution, max pooling, average pooling, or sum pooling.
    """

    def __init__(self, hidden_size: int, kernel_size: int, span_mode: str = 'conv_normal'):
        super().__init__()
        self.span_mode = span_mode
        self.pad = kernel_size - 1

        operation_dict = {
            'conv_conv': nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size),
            'conv_max': nn.MaxPool1d(kernel_size=kernel_size, stride=1),
            'conv_mean': nn.AvgPool1d(kernel_size=kernel_size, stride=1),
            'conv_sum': nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        }
        self.operation = operation_dict.get(span_mode, None)
        if self.operation is None:
            raise ValueError(f"Invalid span_mode: {span_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bld->bdl', x)
        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)
        x = self.operation(x)
        if self.span_mode == "conv_sum":
            x *= (self.pad + 1)
        return torch.einsum('bdl->bld', x)


class SpanConv(nn.Module):
    """
    Applies a sequence of convolution operations to generate span representations.
    """

    def __init__(self, hidden_size: int, max_width: int, span_mode: str, dropout: float = 0.4):
        super().__init__()
        kernels = [i + 2 for i in range(max_width - 1)]
        self.convs = nn.ModuleList([SpanConvBlock(hidden_size, kernel, span_mode) for kernel in kernels])
        self.project = create_projection_layer(hidden_size, dropout)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        span_reps = [x] + [conv(x) for conv in self.convs]
        span_reps = torch.stack(span_reps, dim=-2)
        return self.project(span_reps)


class ConvShare(nn.Module):
    """
    Applies a shared 1D convolution across spans of different widths.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.conv_weights = nn.Parameter(torch.randn(hidden_size, hidden_size, max_width))
        nn.init.xavier_normal_(self.conv_weights)
        self.project_end = create_projection_layer(hidden_size, dropout)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        B, L, D = x.size()
        x = x.permute(0, 2, 1)
        span_reps = [F.conv1d(F.pad(x, (i, 0), "constant", 0), self.conv_weights[:, :, :i + 1]).permute(0, 2, 1)
                     for i in range(self.max_width)]
        span_reps = torch.stack(span_reps, dim=-2)
        return self.project_end(span_reps)


class SpanMarker(nn.Module):
    """
    Marks and projects span endpoints using an MLP.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.span_extractor_start = EndpointSpanExtractor(hidden_size, combination='x')
        self.span_extractor_end = EndpointSpanExtractor(hidden_size, combination='y')
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        start_span_rep = self.span_extractor_start(start_rep, span_idx)
        end_span_rep = self.span_extractor_end(end_rep, span_idx)
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        return self.out_project(cat).view(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """
    Wrapper for various span representation approaches.
    """

    def __init__(self, hidden_size: int, max_width: int, span_mode: str, dropout: float = 0.4):
        super().__init__()

        span_rep_dict = {
            'endpoints': SpanEndpoints(hidden_size, max_width, dropout=dropout),
            'marker': SpanMarker(hidden_size, max_width, dropout),
            'query': SpanQuery(hidden_size, max_width, trainable=True, dropout=dropout),
            'mlp': SpanMLP(hidden_size, max_width, dropout),
            'conv_conv': SpanConv(hidden_size, max_width, span_mode='conv_conv', dropout=dropout),
            'conv_max': SpanConv(hidden_size, max_width, span_mode='conv_max', dropout=dropout),
            'conv_mean': SpanConv(hidden_size, max_width, span_mode='conv_mean', dropout=dropout),
            'conv_sum': SpanConv(hidden_size, max_width, span_mode='conv_sum', dropout=dropout),
            'conv_share': ConvShare(hidden_size, max_width, dropout=dropout)
        }

        self.span_rep_layer = span_rep_dict.get(span_mode, None)
        if self.span_rep_layer is None:
            raise ValueError(f"Invalid span_mode: {span_mode}")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self.span_rep_layer(x, *args)
