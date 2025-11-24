from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


def compute_degree(A: torch.Tensor) -> torch.Tensor:
    """Compute the degree matrix from an adjacency matrix.

    The degree of node i is defined as D_ii = Σ_j A_ij, representing the sum
    of edge weights connected to that node.

    Args:
        A: Adjacency matrix of shape (B, E, E) where B is batch size and E is
            the number of entities/nodes.

    Returns:
        Degree vector of shape (B, E) containing the degree for each node.
        Values are clamped to a minimum of 1e-6 to avoid division by zero.
    """
    return A.sum(dim=-1).clamp(min=1e-6)


def _apply_pair_mask(A: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero out adjacency entries where at least one endpoint is masked.

    This ensures that edges to/from padded entities are properly masked out.
    An edge (i, j) is kept only if both mask[i] and mask[j] are non-zero.

    Args:
        A: Adjacency matrix of shape (B, E, E).
        mask: Optional boolean/float mask of shape (B, E) where 1 indicates
            valid entities and 0 indicates padding. If None, returns A unchanged.

    Returns:
        Masked adjacency matrix of shape (B, E, E).
    """
    if mask is None:
        return A
    m = mask.float()  # (B, E)
    return A * m.unsqueeze(2) * m.unsqueeze(1)  # (B, E, E)


def dot_product_adjacency(
    X: torch.Tensor, mask: Optional[torch.Tensor] = None, normalize: bool = False
) -> torch.Tensor:
    """Compute adjacency matrix using dot-product (cosine) similarity.

    Computes pairwise similarities between entity embeddings using either
    normalized (cosine similarity) or unnormalized dot products, followed
    by sigmoid activation.

    Args:
        X: Entity embeddings of shape (B, E, D) where B is batch size,
            E is number of entities, and D is embedding dimension.
        mask: Optional mask of shape (B, E) indicating valid entities.
        normalize: If True, L2-normalize embeddings before computing similarity
            (results in cosine similarity). Defaults to False.

    Returns:
        Adjacency matrix of shape (B, E, E) with values in (0, 1).
    """
    if normalize:
        Xn = F.normalize(X, p=2, dim=-1)
    else:
        Xn = X
    A = torch.bmm(Xn, Xn.transpose(1, 2))  # (B, E, E)
    A = torch.sigmoid(A)
    return _apply_pair_mask(A, mask)


class MLPDecoder(nn.Module):
    """MLP-based adjacency decoder using concatenated node pairs.

    This decoder concatenates embeddings of node pairs and passes them through
    an MLP to predict edge existence. It models pairwise interactions explicitly.

    Args:
        in_dim: Input embedding dimension.
        hidden_dim: Hidden layer dimension for the MLP.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        """Initialize the MLP decoder.

        Args:
            in_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adjacency matrix using MLP on concatenated node pairs.

        Args:
            X: Entity embeddings of shape (B, E, D).
            mask: Optional mask of shape (B, E) indicating valid entities.

        Returns:
            Adjacency matrix of shape (B, E, E) with values in (0, 1).
        """
        B, E, D = X.shape
        Xi = X.unsqueeze(2).expand(B, E, E, D)
        Xj = X.unsqueeze(1).expand(B, E, E, D)
        A = torch.sigmoid(self.mlp(torch.cat([Xi, Xj], -1)).squeeze(-1))
        return _apply_pair_mask(A, mask)


class AttentionAdjacency(nn.Module):
    """Adjacency matrix derived from multi-head attention weights.

    Uses PyTorch's multi-head attention mechanism to compute pairwise attention
    scores, which are averaged across heads to form the adjacency matrix.

    Args:
        d_model: Model dimension (embedding size).
        nhead: Number of attention heads.
    """

    def __init__(self, d_model: int, nhead: int):
        """Initialize the attention-based adjacency module.

        Args:
            d_model: Model dimension for attention.
            nhead: Number of attention heads.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adjacency matrix from attention weights.

        Args:
            X: Entity embeddings of shape (B, E, D).
            mask: Optional mask of shape (B, E) where 1 indicates valid entities.

        Returns:
            Adjacency matrix of shape (B, E, E) computed from averaged attention weights.
        """
        key_padding = (~mask.bool()) if mask is not None else None
        _, w = self.attn(X, X, X, key_padding_mask=key_padding, need_weights=True)
        if w.dim() == 4:  # (B, h, E, E) - average across heads
            w = w.mean(dim=1)
        w = _apply_pair_mask(w, mask)
        return w


class BilinearDecoder(nn.Module):
    """Bilinear decoder for adjacency prediction.

    Projects embeddings to a latent space and computes adjacency as the
    sigmoid of the bilinear product Z @ Z^T.

    Args:
        in_dim: Input embedding dimension.
        latent_dim: Latent projection dimension.
    """

    def __init__(self, in_dim: int, latent_dim: int):
        """Initialize the bilinear decoder.

        Args:
            in_dim: Input embedding dimension.
            latent_dim: Dimension of the latent projection space.
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adjacency using bilinear projection.

        Args:
            X: Entity embeddings of shape (B, E, D).
            mask: Optional mask of shape (B, E) indicating valid entities.

        Returns:
            Adjacency matrix of shape (B, E, E) with values in (0, 1).
        """
        Z = self.proj(X)
        A = torch.sigmoid(torch.bmm(Z, Z.transpose(1, 2)))
        return _apply_pair_mask(A, mask)


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Network layer with symmetric normalization.

    Implements the GCN propagation rule: H = ReLU(D^(-1/2) A D^(-1/2) X W)
    where D is the degree matrix, A is the adjacency with self-loops, and W
    is a learnable weight matrix.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize the GCN layer.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply graph convolution with symmetric normalization.

        Args:
            X: Node features of shape (B, E, D).
            A: Adjacency matrix of shape (B, E, E).
            mask: Optional mask of shape (B, E). Self-loops are added only
                to valid (non-masked) nodes.

        Returns:
            Updated node features of shape (B, E, out_dim).
        """
        # Keep only valid⇆valid edges & add self-loops on valid nodes
        if mask is not None:
            A = _apply_pair_mask(A, mask)
            A = A + torch.diag_embed(mask.float())  # self-loops only where mask == 1
        else:
            A = A + torch.eye(A.size(1), device=A.device).unsqueeze(0)

        D_inv_sqrt = compute_degree(A).pow(-0.5)
        A_norm = torch.diag_embed(D_inv_sqrt) @ A @ torch.diag_embed(D_inv_sqrt)
        out = torch.bmm(A_norm, X)
        return F.relu(self.linear(out))


class GCNDecoder(nn.Module):
    """GCN-based adjacency decoder.

    First computes an initial adjacency using dot-product similarity, applies
    a GCN layer to update node representations, then predicts the final adjacency
    from the updated representations.

    Args:
        in_dim: Input embedding dimension.
        hidden_dim: Hidden dimension for GCN and projection layers.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        """Initialize the GCN decoder.

        Args:
            in_dim: Input embedding dimension.
            hidden_dim: Hidden dimension for the GCN layer and projection.
        """
        super().__init__()
        self.gcn = SimpleGCNLayer(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adjacency using GCN refinement.

        Args:
            X: Entity embeddings of shape (B, E, D).
            mask: Optional mask of shape (B, E) indicating valid entities.

        Returns:
            Adjacency matrix of shape (B, E, E) with values in (0, 1).
        """
        A0 = dot_product_adjacency(X, mask)  # Initial adjacency (already masked)
        H = self.gcn(X, A0, mask)  # Updated node features
        A = torch.sigmoid(torch.bmm(self.proj(H), self.proj(H).transpose(1, 2)))
        return _apply_pair_mask(A, mask)


class GATDecoder(nn.Module):
    """Graph Attention Network (GAT) based adjacency decoder.

    Uses multi-head attention to update node representations, then predicts
    adjacency from the transformed features.

    Args:
        d_model: Model dimension for attention.
        nhead: Number of attention heads.
        hidden_dim: Hidden dimension for the final projection.
    """

    def __init__(self, d_model: int, nhead: int, hidden_dim: int):
        """Initialize the GAT decoder.

        Args:
            d_model: Model dimension for attention mechanism.
            nhead: Number of attention heads.
            hidden_dim: Hidden dimension for the output projection.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear = nn.Linear(d_model, hidden_dim)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adjacency using GAT refinement.

        Args:
            X: Entity embeddings of shape (B, E, D).
            mask: Optional mask of shape (B, E) indicating valid entities.

        Returns:
            Adjacency matrix of shape (B, E, E) with values in (0, 1).
        """
        key_padding = (~mask.bool()) if mask is not None else None
        H, w = self.attn(X, X, X, key_padding_mask=key_padding, need_weights=True)
        if w.dim() == 4:
            w = w.mean(dim=1)  # (B, E, E) - average across heads
        Z = self.linear(H)
        A = torch.sigmoid(torch.bmm(Z, Z.transpose(1, 2)))
        return _apply_pair_mask(A, mask)


class RelationsRepLayer(nn.Module):
    """Unified wrapper for different adjacency computation methods.

    This layer provides a common interface for various approaches to computing
    adjacency matrices from entity embeddings, including:
    - 'dot': Dot-product/cosine similarity
    - 'mlp': MLP-based pairwise decoder
    - 'attention'/'attn': Multi-head attention weights
    - 'bilinear': Bilinear projection
    - 'gcn': Graph convolutional refinement
    - 'gat': Graph attention network

    All methods support masked inputs for handling variable-length sequences.

    Args:
        in_dim: Input embedding dimension.
        relation_mode: String specifying the adjacency computation method.
            One of: 'dot', 'mlp', 'attention', 'attn', 'bilinear', 'gcn', 'gat'.
        **kwargs: Additional arguments passed to specific decoders:
            - hidden_dim (int): For 'mlp', 'gcn', 'gat'. Defaults to in_dim.
            - nhead (int): For 'attention'/'attn' and 'gat'. Defaults to 8.
            - latent_dim (int): For 'bilinear'. Defaults to in_dim.

    Raises:
        ValueError: If relation_mode is not one of the supported methods.

    Example:
        >>> layer = RelationsRepLayer(in_dim=128, relation_mode="gcn", hidden_dim=64)
        >>> X = torch.randn(4, 10, 128)  # (batch=4, entities=10, dim=128)
        >>> mask = torch.ones(4, 10)  # All entities valid
        >>> A = layer(X, mask)  # (4, 10, 10) adjacency matrix
    """

    def __init__(self, in_dim: int, relation_mode: str, **kwargs: Any):
        """Initialize the relations representation layer.

        Args:
            in_dim: Input embedding dimension.
            relation_mode: Adjacency computation method. One of: 'dot', 'mlp',
                'attention', 'attn', 'bilinear', 'gcn', 'gat'.
            **kwargs: Method-specific arguments (hidden_dim, nhead, latent_dim).

        Raises:
            ValueError: If relation_mode is not recognized.
        """
        super().__init__()
        m = relation_mode.lower()

        if m == "dot":

            class _Dot(nn.Module):
                """Simple wrapper for dot-product adjacency with mask support."""

                def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                    return dot_product_adjacency(X, mask)

            self.relation_rep_layer = _Dot()

        elif m == "mlp":
            self.relation_rep_layer = MLPDecoder(in_dim, kwargs.get("hidden_dim", in_dim))

        elif m in {"attention", "attn"}:
            self.relation_rep_layer = AttentionAdjacency(in_dim, kwargs.get("nhead", 8))

        elif m == "bilinear":
            self.relation_rep_layer = BilinearDecoder(in_dim, kwargs.get("latent_dim", in_dim))

        elif m == "gcn":
            self.relation_rep_layer = GCNDecoder(in_dim, kwargs.get("hidden_dim", in_dim))

        elif m == "gat":
            self.relation_rep_layer = GATDecoder(in_dim, kwargs.get("nhead", 8), kwargs.get("hidden_dim", in_dim))
        else:
            raise ValueError(f"Unknown relation mode: {relation_mode}")

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute adjacency matrix from entity embeddings.

        Args:
            X: Entity/mention embeddings of shape (B, E, D) where B is batch size,
                E is number of entities, and D is embedding dimension.
            mask: Optional mask of shape (B, E) where 1 indicates valid entities
                and 0 indicates padding.
            *args: Additional positional arguments (unused, for compatibility).
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            Adjacency matrix of shape (B, E, E) with values in [0, 1].
            Entries A[b, i, j] represent the predicted edge weight from
            entity i to entity j in batch b.
        """
        return self.relation_rep_layer(X, *args, mask=mask, **kwargs)
