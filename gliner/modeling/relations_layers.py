import torch, torch.nn.functional as F
from torch import nn

def compute_degree(A: torch.Tensor):
    """Degree matrix: D_ii = Σ_j A_ij   (B, E)"""
    return A.sum(dim=-1).clamp(min=1e-6)          # avoid /0

def _apply_pair_mask(A: torch.Tensor, mask: torch.Tensor | None):
    """Zero out entries where at least one endpoint is masked."""
    if mask is None:
        return A
    m = mask.float()                              # (B, E)
    return A * m.unsqueeze(2) * m.unsqueeze(1)    # (B, E, E)

def cosine_adjacency(X: torch.Tensor, mask: torch.Tensor | None = None):
    # X : (B, E, D)
    Xn = F.normalize(X, p=2, dim=-1)
    A  = torch.bmm(Xn, Xn.transpose(1, 2))        # cos-sim
    A  = torch.sigmoid(A)
    return _apply_pair_mask(A, mask)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, mask: torch.Tensor | None = None):
        B, E, D = X.shape
        Xi = X.unsqueeze(2).expand(B, E, E, D)
        Xj = X.unsqueeze(1).expand(B, E, E, D)
        A  = torch.sigmoid(self.mlp(torch.cat([Xi, Xj], -1)).squeeze(-1))
        return _apply_pair_mask(A, mask)


class AttentionAdjacency(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, X, mask: torch.Tensor | None = None):
        key_padding = (~mask.bool()) if mask is not None else None
        _, w = self.attn(X, X, X,
                         key_padding_mask=key_padding,
                         need_weights=True)       # (B, h, E, E)
        if w.dim() == 4:                          # average heads
            w = w.mean(dim=1)
        w = _apply_pair_mask(w, mask)
        return w


class BilinearDecoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, X, mask: torch.Tensor | None = None):
        Z  = self.proj(X)
        A  = torch.sigmoid(torch.bmm(Z, Z.transpose(1, 2)))
        return _apply_pair_mask(A, mask)


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A, mask: torch.Tensor | None = None):
        # keep only valid⇆valid edges & add self-loops on valid nodes
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
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gcn  = SimpleGCNLayer(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, mask: torch.Tensor | None = None):
        A0 = cosine_adjacency(X, mask)            # already masked
        H  = self.gcn(X, A0, mask)
        A  = torch.sigmoid(torch.bmm(self.proj(H), self.proj(H).transpose(1, 2)))
        return _apply_pair_mask(A, mask)


class GATDecoder(nn.Module):
    def __init__(self, d_model, nhead, hidden_dim):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear = nn.Linear(d_model, hidden_dim)

    def forward(self, X, mask: torch.Tensor | None = None):
        key_padding = (~mask.bool()) if mask is not None else None
        H, w = self.attn(X, X, X,
                         key_padding_mask=key_padding,
                         need_weights=True)
        if w.dim() == 4:
            w = w.mean(dim=1)                     # (B, E, E)
        Z = self.linear(H)
        A = torch.sigmoid(torch.bmm(Z, Z.transpose(1, 2)))
        return _apply_pair_mask(A, mask)

class RelationsRepLayer(nn.Module):
    """
    Wrapper around different adjacency builders.
    All decoders now accept a boolean/float mask of shape (B, E).
    """

    def __init__(self, in_dim: int, relation_mode: str, **kwargs):
        super().__init__()
        m = relation_mode.lower()

        if m == 'cosine':
            class _Cos(nn.Module):
                def forward(_, X, mask=None):       # tiny wrapper for mask
                    return cosine_adjacency(X, mask)
            self.relation_rep_layer = _Cos()

        elif m == 'mlp':
            self.relation_rep_layer = MLPDecoder(in_dim, kwargs.get('hidden_dim', in_dim))

        elif m in {'attention', 'attn'}:
            self.relation_rep_layer = AttentionAdjacency(in_dim, kwargs.get('nhead', 8))

        elif m == 'bilinear':
            self.relation_rep_layer = BilinearDecoder(in_dim, kwargs.get('latent_dim', in_dim))

        elif m == 'gcn':
            self.relation_rep_layer = GCNDecoder(in_dim, kwargs.get('hidden_dim', in_dim))

        elif m == 'gat':
            self.relation_rep_layer = GATDecoder(in_dim,
                                                 kwargs.get('nhead', 8),
                                                 kwargs.get('hidden_dim', in_dim))
        else:
            raise ValueError(f"Unknown relation mode: {relation_mode}")

    def forward(self, X, mask: torch.Tensor | None = None, *args, **kw):
        """
        X    : (B, E, D) entity/mention embeddings
        mask : (B, E)    1 = valid, 0 = padding  (optional)
        """
        return self.relation_rep_layer(X, mask=mask, *args, **kw)
