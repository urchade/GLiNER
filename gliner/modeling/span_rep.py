import torch
import torch.nn.functional as F
from torch import nn

from .layers import create_projection_layer


class SpanQuery(nn.Module):
    """Span representation using learned query vectors.

    This layer learns a set of query vectors, one for each span width, and
    projects token representations onto these queries to produce span
    representations.

    Attributes:
        query_seg (nn.Parameter): Learnable query matrix of shape [hidden_size, max_width].
        project (nn.Sequential): MLP projection layer with ReLU activation.
    """

    def __init__(self, hidden_size, max_width, trainable=True):
        """Initialize the SpanQuery layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            trainable (bool, optional): Whether query parameters are trainable.
                Defaults to True.
        """
        super().__init__()

        self.query_seg = nn.Parameter(torch.randn(hidden_size, max_width))

        nn.init.uniform_(self.query_seg, a=-1, b=1)

        if not trainable:
            self.query_seg.requires_grad = False

        self.project = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

    def forward(self, h, *args):
        """Compute span representations using query projection.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        span_rep = torch.einsum("bld, ds->blsd", h, self.query_seg)

        return self.project(span_rep)


class SpanMLP(nn.Module):
    """Span representation using a simple MLP.

    This layer applies a linear transformation to produce multiple span
    representations per position.

    Attributes:
        mlp (nn.Linear): Linear layer that expands hidden_size to
            hidden_size * max_width.
    """

    def __init__(self, hidden_size, max_width):
        """Initialize the SpanMLP layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
        """
        super().__init__()

        self.max_width = max_width
        self.mlp = nn.Linear(hidden_size, hidden_size * max_width)

    def forward(self, h, *args):
        """Compute span representations using MLP projection.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D]
                with ReLU activation applied.
        """
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        span_rep = self.mlp(h)

        span_rep = span_rep.view(B, L, self.max_width, D)

        return span_rep.relu()


class SpanCAT(nn.Module):
    """Span representation using concatenation with learned queries.

    This layer concatenates token representations with learnable query vectors
    and projects them to produce span representations.

    Attributes:
        max_width (int): Maximum span width to represent.
        query_seg (nn.Parameter): Learnable query matrix of shape [128, max_width].
        project (nn.Sequential): MLP projection layer with ReLU activation.
    """

    def __init__(self, hidden_size, max_width):
        """Initialize the SpanCAT layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
        """
        super().__init__()

        self.max_width = max_width

        self.query_seg = nn.Parameter(torch.randn(128, max_width))

        self.project = nn.Sequential(nn.Linear(hidden_size + 128, hidden_size), nn.ReLU())

    def forward(self, h, *args):
        """Compute span representations by concatenating with queries.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        # h of shape [B, L, D]
        # query_seg of shape [D, max_width]

        B, L, D = h.size()

        h = h.view(B, L, 1, D).repeat(1, 1, self.max_width, 1)

        q = self.query_seg.view(1, 1, self.max_width, -1).repeat(B, L, 1, 1)

        span_rep = torch.cat([h, q], dim=-1)

        span_rep = self.project(span_rep)

        return span_rep


class SpanConvBlock(nn.Module):
    """A single convolutional block for span representation.

    This block applies either convolution or pooling operations with a specific
    kernel size to capture span information.

    Attributes:
        conv (nn.Module): Convolution or pooling layer.
        span_mode (str): Type of operation ('conv_conv', 'conv_max', 'conv_mean', 'conv_sum').
        pad (int): Padding size for the operation.
    """

    def __init__(self, hidden_size, kernel_size, span_mode="conv_normal"):
        """Initialize the SpanConvBlock.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            kernel_size (int): Size of the convolution/pooling kernel.
            span_mode (str, optional): Type of operation to use. Options are:
                'conv_conv', 'conv_max', 'conv_mean', 'conv_sum'.
                Defaults to 'conv_normal'.
        """
        super().__init__()

        if span_mode == "conv_conv":
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size)

            # initialize the weights
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")

        elif span_mode == "conv_max":
            self.conv = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        elif span_mode in {"conv_mean", "conv_sum"}:
            self.conv = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

        self.span_mode = span_mode

        self.pad = kernel_size - 1

    def forward(self, x):
        """Apply the convolutional block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].

        Returns:
            torch.Tensor: Output tensor of shape [B, L, D].
        """
        x = torch.einsum("bld->bdl", x)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad), "constant", 0)

        x = self.conv(x)

        if self.span_mode == "conv_sum":
            x = x * (self.pad + 1)

        return torch.einsum("bdl->bld", x)


class SpanConv(nn.Module):
    """Span representation using multiple convolutional layers.

    This layer uses convolutions with different kernel sizes to capture
    spans of different widths.

    Attributes:
        convs (nn.ModuleList): List of convolutional blocks with varying kernel sizes.
        project (nn.Sequential): MLP projection layer with ReLU activation.
    """

    def __init__(self, hidden_size, max_width, span_mode):
        """Initialize the SpanConv layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            span_mode (str): Type of convolution operation to use.
        """
        super().__init__()

        kernels = [i + 2 for i in range(max_width - 1)]

        self.convs = nn.ModuleList()

        for kernel in kernels:
            self.convs.append(SpanConvBlock(hidden_size, kernel, span_mode))

        self.project = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x, *args):
        """Compute span representations using multiple convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        span_reps = [x]

        for conv in self.convs:
            h = conv(x)
            span_reps.append(h)

        span_reps = torch.stack(span_reps, dim=-2)

        return self.project(span_reps)


class SpanEndpointsBlock(nn.Module):
    """Extract start and end token representations for spans.

    This block extracts the first and last token of each span.

    Attributes:
        kernel_size (int): The span width (kernel size).
    """

    def __init__(self, kernel_size):
        """Initialize the SpanEndpointsBlock.

        Args:
            kernel_size (int): The span width to extract endpoints for.
        """
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        """Extract start and end representations for all spans.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].

        Returns:
            torch.Tensor: Start and end representations of shape [B, L, 2, D].
        """
        B, L, D = x.size()

        span_idx = torch.LongTensor([[i, i + self.kernel_size - 1] for i in range(L)]).to(x.device)

        x = F.pad(x, (0, 0, 0, self.kernel_size - 1), "constant", 0)

        # endrep
        start_end_rep = torch.index_select(x, dim=1, index=span_idx.view(-1))

        start_end_rep = start_end_rep.view(B, L, 2, D)

        return start_end_rep


class ConvShare(nn.Module):
    """Span representation using shared convolution weights.

    This layer uses a single set of convolution weights shared across
    different span widths.

    Attributes:
        max_width (int): Maximum span width to represent.
        conv_weigth (nn.Parameter): Shared convolution weights of shape
            [hidden_size, hidden_size, max_width].
        project (nn.Sequential): MLP projection layer with ReLU activation.
    """

    def __init__(self, hidden_size, max_width):
        """Initialize the ConvShare layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
        """
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(torch.randn(hidden_size, hidden_size, max_width))

        nn.init.kaiming_uniform_(self.conv_weigth, nonlinearity="relu")

        self.project = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, x, *args):
        """Compute span representations using shared convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        span_reps = []

        x = torch.einsum("bld->bdl", x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, : i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        out = torch.stack(span_reps, dim=-2)

        return self.project(out)


def extract_elements(sequence, indices):
    """Extract elements from a sequence using provided indices.

    Args:
        sequence (torch.Tensor): Input sequence of shape [B, L, D].
        indices (torch.Tensor): Indices to extract, shape [B, K].

    Returns:
        torch.Tensor: Extracted elements of shape [B, K, D].
    """
    B, L, D = sequence.size()
    indices = torch.clamp(indices, 0, L - 1)
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)
    extracted_elements = torch.gather(sequence, 1, expanded_indices)
    return extracted_elements


class SpanMarker(nn.Module):
    """Span representation using marker-based approach.

    This layer projects start and end positions separately and combines them
    to form span representations.

    Attributes:
        max_width (int): Maximum span width to represent.
        project_start (nn.Sequential): MLP for projecting start positions.
        project_end (nn.Sequential): MLP for projecting end positions.
        out_project (nn.Linear): Final projection layer.
    """

    def __init__(self, hidden_size, max_width, dropout=0.4):
        """Initialize the SpanMarker layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            dropout (float, optional): Dropout rate. Defaults to 0.4.
        """
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
        """Compute span representations using start and end markers.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            span_idx (torch.Tensor): Span indices of shape [B, *, 2] where
                span_idx[..., 0] are start indices and span_idx[..., 1] are
                end indices.

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
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
    """Marks and projects span endpoints using an MLP.

    A cleaner version of SpanMarker using the create_projection_layer utility.

    Attributes:
        max_width (int): Maximum span width to represent.
        project_start (nn.Module): MLP for projecting start positions.
        project_end (nn.Module): MLP for projecting end positions.
        out_project (nn.Module): Final projection layer.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        """Initialize the SpanMarkerV0 layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            dropout (float, optional): Dropout rate. Defaults to 0.4.
        """
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """Compute span representations using start and end markers.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            span_idx (torch.Tensor): Span indices of shape [B, *, 2].

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        B, L, D = h.size()

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat).view(B, L, self.max_width, D)


class SpanMarkerV1(nn.Module):
    """Marks span endpoints and augments them with the first-token embedding.

    For each candidate span we build
        [ start_proj ‖ end_proj ‖ first_token_proj ]  →  MLP  → span_rep
    and finally reshape to [B, L, max_width, D].

    Attributes:
        max_width (int): Maximum span width to represent.
        project_start (nn.Module): MLP for projecting start positions.
        project_end (nn.Module): MLP for projecting end positions.
        project_first (nn.Module): MLP for projecting the average token.
        out_project (nn.Module): Final projection layer.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        """Initialize the SpanMarkerV1 layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            dropout (float, optional): Dropout rate. Defaults to 0.4.
        """
        super().__init__()
        self.max_width = max_width

        # Independent projections for the three ingredients
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.project_first = create_projection_layer(hidden_size, dropout)

        # 3 x hidden_size (start + end + first)  →  hidden_size
        self.out_project = create_projection_layer(hidden_size * 3, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """Compute span representations with average token augmentation.

        For each span, concatenates start marker, end marker, and average
        token embedding, then projects to produce the final representation.

        Args:
            h (torch.Tensor): Token representations, shape [B, L, D].
            span_idx (torch.Tensor): Indices of candidate spans, shape [B, *, 2]
                (* can be L x max_width or any flattened span dimension).

        Returns:
            torch.Tensor: Span representations, shape [B, L, max_width, D].
        """
        B, L, D = h.size()

        # Pre-compute per-token projections
        start_rep = self.project_start(h)  # [B, L, D]
        end_rep = self.project_end(h)  # [B, L, D]

        # Project the first-token embedding once
        average_token_proj = torch.mean(h, dim=1)

        # Gather start/end representations for each span
        start_span_rep = extract_elements(start_rep, span_idx[..., 0])  # [B, S, D]
        end_span_rep = extract_elements(end_rep, span_idx[..., 1])  # [B, S, D]

        # Broadcast first-token embedding to every span
        first_span_rep = average_token_proj.unsqueeze(1).expand_as(start_span_rep)  # [B, S, D]

        # Concatenate and project
        span_feat = torch.cat((start_span_rep, end_span_rep, first_span_rep), dim=-1).relu()  # [B, S, 3D]

        out = self.out_project(span_feat)  # [B, S, D]

        # Reshape back to [B, L, max_width, D] (S = L x max_width)
        return out.view(B, L, self.max_width, D)


class ConvShareV2(nn.Module):
    """Span representation using shared convolution weights (version 2).

    Similar to ConvShare but uses Xavier initialization and no projection layer.

    Attributes:
        max_width (int): Maximum span width to represent.
        conv_weigth (nn.Parameter): Shared convolution weights of shape
            [hidden_size, hidden_size, max_width].
    """

    def __init__(self, hidden_size, max_width):
        """Initialize the ConvShareV2 layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
        """
        super().__init__()

        self.max_width = max_width

        self.conv_weigth = nn.Parameter(torch.randn(hidden_size, hidden_size, max_width))

        nn.init.xavier_normal_(self.conv_weigth)

    def forward(self, x, *args):
        """Compute span representations using shared convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            *args: Additional arguments (unused).

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        span_reps = []

        x = torch.einsum("bld->bdl", x)

        for i in range(self.max_width):
            pad = i
            x_i = F.pad(x, (0, pad), "constant", 0)
            conv_w = self.conv_weigth[:, :, : i + 1]
            out_i = F.conv1d(x_i, conv_w)
            span_reps.append(out_i.transpose(-1, -2))

        out = torch.stack(span_reps, dim=-2)

        return out


class TokenMarker(nn.Module):
    """Marks and projects span endpoints using an MLP.

    A cleaner version of SpanMarker using the create_projection_layer utility.

    Attributes:
        max_width (int): Maximum span width to represent.
        project_start (nn.Module): MLP for projecting start positions.
        project_end (nn.Module): MLP for projecting end positions.
        out_project (nn.Module): Final projection layer.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.4):
        """Initialize the SpanMarkerV0 layer.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            dropout (float, optional): Dropout rate. Defaults to 0.4.
        """
        super().__init__()
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        """Compute span representations using start and end markers.

        Args:
            h (torch.Tensor): Token representations of shape [B, L, D].
            span_idx (torch.Tensor): Span indices of shape [B, *, 2].

        Returns:
            torch.Tensor: Span representations of shape [B, L, max_width, D].
        """
        B, L, D = h.size()
        num_spans = span_idx.size(1)
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat)


class SpanRepLayer(nn.Module):
    """Factory class for various span representation approaches.

    This class provides a unified interface to instantiate different span
    representation methods based on the specified mode.

    Attributes:
        span_rep_layer (nn.Module): The underlying span representation layer.
    """

    def __init__(self, hidden_size, max_width, span_mode, **kwargs):
        """Initialize the SpanRepLayer with the specified mode.

        Args:
            hidden_size (int): Dimension of the hidden representations.
            max_width (int): Maximum span width to represent.
            span_mode (str): Type of span representation to use. Options:
                - 'marker': SpanMarker
                - 'markerV0': SpanMarkerV0
                - 'markerV1': SpanMarkerV1
                - 'query': SpanQuery
                - 'mlp': SpanMLP
                - 'cat': SpanCAT
                - 'conv_conv': SpanConv with convolution
                - 'conv_max': SpanConv with max pooling
                - 'conv_mean': SpanConv with mean pooling
                - 'conv_sum': SpanConv with sum pooling
                - 'conv_share': ConvShare
            **kwargs: Additional arguments passed to the span representation layer.

        Raises:
            ValueError: If an unknown span_mode is provided.
        """
        super().__init__()

        if span_mode == "marker":
            self.span_rep_layer = SpanMarker(hidden_size, max_width, **kwargs)
        elif span_mode == "markerV0":
            self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)
        elif span_mode == "markerV1":
            self.span_rep_layer = SpanMarkerV1(hidden_size, max_width, **kwargs)
        elif span_mode == "query":
            self.span_rep_layer = SpanQuery(hidden_size, max_width, trainable=True)
        elif span_mode == "mlp":
            self.span_rep_layer = SpanMLP(hidden_size, max_width)
        elif span_mode == "cat":
            self.span_rep_layer = SpanCAT(hidden_size, max_width)
        elif span_mode == "conv_conv":
            self.span_rep_layer = SpanConv(hidden_size, max_width, span_mode="conv_conv")
        elif span_mode == "conv_max":
            self.span_rep_layer = SpanConv(hidden_size, max_width, span_mode="conv_max")
        elif span_mode == "conv_mean":
            self.span_rep_layer = SpanConv(hidden_size, max_width, span_mode="conv_mean")
        elif span_mode == "conv_sum":
            self.span_rep_layer = SpanConv(hidden_size, max_width, span_mode="conv_sum")
        elif span_mode == "conv_share":
            self.span_rep_layer = ConvShare(hidden_size, max_width)
        elif span_mode == "token_level":
            self.span_rep_layer = TokenMarker(hidden_size, **kwargs)
        else:
            raise ValueError(f"Unknown span mode {span_mode}")

    def forward(self, x, *args):
        """Forward pass through the selected span representation layer.

        Args:
            x (torch.Tensor): Input tensor, typically of shape [B, L, D].
            *args: Additional arguments passed to the underlying layer.

        Returns:
            torch.Tensor: Span representations, typically of shape
                [B, L, max_width, D].
        """
        return self.span_rep_layer(x, *args)
