from __future__ import annotations

import torch
from torch import nn, fft
from torch.nn import functional as F

from ..layers import create_projection_layer


def _split_complex(x):
    """Assume last dim = 2k and split into real / imag parts."""
    return torch.chunk(x, 2, dim=-1)


def _split_quaternion(x):
    """Assume last dim = 4k and split into (1,i,j,k) parts."""
    return torch.chunk(x, 4, dim=-1)


def _norm_clamp(x, max_norm):
    return x.clamp(max=max_norm) if max_norm is not None else x


class NormBasedInteraction(nn.Module):
    def __init__(
        self,
        dim: int,
        p: int = 2,
        power: float = 1.0,
        clamp_norm: float | None = 10.0,
        use_scorer: bool = False,
        dropout: float = 0.3,
    ):
        """
        Base class for norm-based KGE interactions.

        Args:
            dim: Embedding dimension
            p: ℓ_p norm (e.g. 1 or 2) used in ‖·‖_p
            power: Raise norm to this power before negating
            clamp_norm: Optional upper bound for numerical stability
            use_scorer: If True, use learned projection instead of norm
            dropout: Dropout rate for scorer (if used)
        """
        super().__init__()
        self.p = p
        self.power = power
        self.clamp = clamp_norm

        # Optional learned scorer instead of norm
        if use_scorer:
            self.scorer = create_projection_layer(dim, dropout, 1)
        else:
            self.scorer = None

    def _score(self, x):
        """
        Score residual vector.

        Args:
            x: (..., D) residual vector

        Returns:
            scores: (...) scalar scores (higher = better)
        """
        if self.scorer is not None:
            # Use learned projection
            return self.scorer(x).squeeze(-1)

        # Use norm-based scoring
        d = torch.linalg.norm(x, ord=self.p, dim=-1)
        d = d.pow(self.power)
        d = _norm_clamp(d, self.clamp)
        return -d  # Negative distance (higher = better)


class UMInteraction(NormBasedInteraction):
    """Unstructured model ‖h - t‖."""

    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def forward(self, h, r, t):
        return self._score(h - t)


class SEInteraction(NormBasedInteraction):
    """Structure Embedding (SE).

    Uses head / tail specific diagonal matrices built from relation.
    h' = diag(r) · h    ,    t' = diag(r) · t
    """

    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def forward(self, h, r, t):
        diag = torch.diag_embed(r)  # (..., D, D)
        h_ = torch.matmul(diag, h.unsqueeze(-1)).squeeze(-1)
        t_ = torch.matmul(diag, t.unsqueeze(-1)).squeeze(-1)
        return self._score(h_ - t_)


class TransEInteraction(NormBasedInteraction):
    """TransE ‖h + r − t‖."""

    def __init__(self, dim: int = 768, p: int = 1, **kwargs):
        super().__init__(dim=dim, p=p, **kwargs)

    def forward(self, h, r, t):
        return self._score(h + r - t)


class TransHInteraction(NormBasedInteraction):
    """
    TransH – project entities to a relation-specific hyperplane.

    Learn mappings from base relation r to:
        r_tr = W_tr * r + b_tr  (translation)
        w    = W_w  * r + b_w   (hyperplane normal)
    """

    def __init__(self, dim: int, p: int = 2, power: float = 1.0, **kwargs):
        super().__init__(dim=dim, p=p, power=power, **kwargs)
        self.r_to_rtr = nn.Linear(dim, dim)
        self.r_to_w = nn.Linear(dim, dim)

    def forward(self, h, r, t):
        # Map base relation vector -> translation & normal
        r_tr = self.r_to_rtr(r)  # (..., D)
        w = self.r_to_w(r)  # (..., D)
        w = F.normalize(w, dim=-1)

        def proj(x):
            # Project: x_proj = x - (x·w) w
            dot = (x * w).sum(dim=-1, keepdim=True)
            return x - dot * w

        h_proj = proj(h)
        t_proj = proj(t)
        return self._score(h_proj + r_tr - t_proj)


class TransFInteraction(NormBasedInteraction):
    """
    TransF – element-wise relation-specific scaling before translation.

    Learn mappings from base relation r to:
        r_vec = W_r * r + b_r
        alpha = W_alpha * r + b_alpha
        beta  = W_beta * r + b_beta

    Score is ‖(alpha ∘ h) + r_vec − (beta ∘ t)‖_p
    """

    def __init__(self, dim: int, p: int = 2, power: float = 1.0, **kwargs):
        super().__init__(dim=dim, p=p, power=power, **kwargs)
        self.r_to_rvec = nn.Linear(dim, dim)
        self.r_to_alpha = nn.Linear(dim, dim)
        self.r_to_beta = nn.Linear(dim, dim)

        # Initialize to start close to plain TransE
        # r_vec ≈ r (identity), alpha ≈ 1, beta ≈ 1
        if dim == self.r_to_rvec.weight.shape[1]:
            nn.init.eye_(self.r_to_rvec.weight)
        else:
            nn.init.xavier_uniform_(self.r_to_rvec.weight)
        nn.init.zeros_(self.r_to_rvec.bias)

        nn.init.zeros_(self.r_to_alpha.weight)
        nn.init.ones_(self.r_to_alpha.bias)

        nn.init.zeros_(self.r_to_beta.weight)
        nn.init.ones_(self.r_to_beta.bias)

    def forward(self, h, r, t):
        r_vec = self.r_to_rvec(r)  # (..., D)
        alpha = self.r_to_alpha(r)  # (..., D)
        beta = self.r_to_beta(r)  # (..., D)

        h_ = alpha * h
        t_ = beta * t
        return self._score(h_ + r_vec - t_)


class PairREInteraction(NormBasedInteraction):
    """
    PairRE – per-relation element-wise scaling of h & t.

    Learn mappings from base relation r to:
        alpha = W_alpha * r + b_alpha
        beta  = W_beta * r + b_beta
    """

    def __init__(self, dim: int, p: int = 2, power: float = 1.0, **kwargs):
        super().__init__(dim=dim, p=p, power=power, **kwargs)
        self.r_to_alpha = nn.Linear(dim, dim)
        self.r_to_beta = nn.Linear(dim, dim)

    def forward(self, h, r, t):
        alpha = self.r_to_alpha(r)  # (..., D)
        beta = self.r_to_beta(r)  # (..., D)
        return self._score(alpha * h - beta * t)


class TripleREInteraction(NormBasedInteraction):
    """
    TripleRE – LineaRE + scalar γ per relation.

    Learn mappings from base relation r to:
        alpha = W_alpha * r + b_alpha
        beta  = W_beta * r + b_beta
        delta = W_delta * r + b_delta
        gamma = w_gamma^T * r + b_gamma (scalar)
    """

    def __init__(self, dim: int, p: int = 2, power: float = 1.0, **kwargs):
        super().__init__(dim=dim, p=p, power=power, **kwargs)
        self.r_to_alpha = nn.Linear(dim, dim)
        self.r_to_beta = nn.Linear(dim, dim)
        self.r_to_delta = nn.Linear(dim, dim)
        self.r_to_gamma = nn.Linear(dim, 1)

    def forward(self, h, r, t):
        alpha = self.r_to_alpha(r)  # (..., D)
        beta = self.r_to_beta(r)  # (..., D)
        delta = self.r_to_delta(r)  # (..., D)
        gamma = self.r_to_gamma(r)  # (..., 1)

        base_score = self._score(alpha * h + delta - beta * t)  # (...)
        return gamma.squeeze(-1) * base_score


class DistMultInteraction(nn.Module):
    """DistMult – Σ_d h_d r_d t_d."""

    def forward(self, h, r, t):
        return (h * r * t).sum(dim=-1)


class SimplEInteraction(nn.Module):
    """SimplE – split every embedding into (forward, backward) halves.

    score = ½( ⟨h_f, r_f, t_b⟩ + ⟨t_f, r_b, h_b⟩ )
    Requires even dimension.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SimplE requires even dimension, got {dim}")

    def forward(self, h, r, t):
        h_f, h_b = _split_complex(h)
        t_f, t_b = _split_complex(t)
        r_f, r_b = _split_complex(r)
        s1 = (h_f * r_f * t_b).sum(dim=-1)
        s2 = (t_f * r_b * h_b).sum(dim=-1)
        return 0.5 * (s1 + s2)


class TuckERInteraction(nn.Module):
    """
    TuckER – global core tensor W (D_r × D_e × D_e).
    """

    def __init__(self, d_e: int, d_r: int, dropout: float = 0.2):
        super().__init__()
        self.d_e = d_e
        self.d_r = d_r
        self.W = nn.Parameter(torch.empty(d_r, d_e, d_e))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(d_e)
        self.bn1 = nn.BatchNorm1d(d_e)
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, h, r, t):
        # Store original shape for reshaping later
        orig_shape = h.shape[:-1]

        # Flatten to 2D for BatchNorm: (batch_size, d_e)
        h_2d = h.reshape(-1, self.d_e)
        t_2d = t.reshape(-1, self.d_e)

        # Apply BatchNorm and dropout
        h_bn = self.bn0(h_2d)
        t_bn = self.bn1(t_2d)

        # Reshape back to original shape (except last dim)
        h_bn = h_bn.view(*orig_shape, self.d_e)
        t_bn = t_bn.view(*orig_shape, self.d_e)

        # Apply input dropout
        h_bn = self.input_dropout(h_bn)
        t_bn = self.input_dropout(t_bn)

        # Reshape r for matrix multiplication
        r_shape = r.shape[:-1]
        r_2d = r.reshape(-1, self.d_r)

        # Core interaction: r x W → (batch, d_e, d_e)
        W_mat = torch.matmul(r_2d, self.W)  # (batch, d_e, d_e)
        W_mat = W_mat.view(*r_shape, self.d_e, self.d_e)

        # Apply dropout on core tensor output
        W_mat = self.dropout(W_mat)

        # Compute h x W_mat
        hr = torch.matmul(h_bn.unsqueeze(-2), W_mat).squeeze(-2)

        # Final score
        scores = (hr * t_bn).sum(dim=-1)
        return scores


class DistMAInteraction(nn.Module):
    """DistMA – sum of pairwise dot products."""

    def forward(self, h, r, t):
        return (h * r).sum(dim=-1) + (h * t).sum(dim=-1) + (r * t).sum(dim=-1)


class ComplExInteraction(nn.Module):
    """ComplEx – Re(⟨h, r, conj(t)⟩) with complex embeddings.

    Requires even dimension.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"ComplEx requires even dimension, got {dim}")

    def forward(self, h, r, t):
        h_re, h_im = _split_complex(h)
        r_re, r_im = _split_complex(r)
        t_re, t_im = _split_complex(t)
        return (h_re * r_re * t_re + h_re * r_im * t_im + h_im * r_re * t_im - h_im * r_im * t_re).sum(dim=-1)


class QuatEInteraction(nn.Module):
    """QuatE – use Hamilton product (a,b,c,d)⨂(e,f,g,h).

    Requires dimension divisible by 4.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"QuatE requires dimension divisible by 4, got {dim}")

    def forward(self, h, r, t):
        h0, h1, h2, h3 = _split_quaternion(h)
        r0, r1, r2, r3 = _split_quaternion(r)
        t0, t1, t2, t3 = _split_quaternion(t)
        # Hamilton product h ⨂ r
        A0 = h0 * r0 - h1 * r1 - h2 * r2 - h3 * r3
        A1 = h0 * r1 + h1 * r0 + h2 * r3 - h3 * r2
        A2 = h0 * r2 - h1 * r3 + h2 * r0 + h3 * r1
        A3 = h0 * r3 + h1 * r2 - h2 * r1 + h3 * r0
        return (A0 * t0 + A1 * t1 + A2 * t2 + A3 * t3).sum(dim=-1)


class HolEInteraction(nn.Module):
    """HolE – circular correlation ϕ(h, t) · r."""

    def forward(self, h, r, t):
        # Convert to float32 for FFT stability
        h = h.to(torch.float32)
        r = r.to(torch.float32)
        t = t.to(torch.float32)

        # FFT-based circular correlation
        fft_h = fft.rfft(h, dim=-1)
        fft_t = fft.rfft(t, dim=-1)
        corr = fft.irfft(fft_h.conj() * fft_t, n=h.shape[-1], dim=-1)
        return (corr * r).sum(dim=-1)


class ERMLPInteraction(nn.Module):
    """ER-MLP: 2-layer perceptron on concatenated [h, r, t]."""

    def __init__(self, dim: int, hidden: int = 2048):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=-1)
        return self.mlp(x).squeeze(-1)


class ConvKBInteraction(nn.Module):
    """ConvKB: Convolutional Knowledge Base interaction (Conv1d version)."""

    def __init__(self, dim: int, n_filters: int = 32, dropout: float = 0.3, use_bias: bool = True):
        super().__init__()

        self.dim = dim
        self.n_filters = n_filters

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Conv1d over the 3-embedding dimension
        self.conv = nn.Conv1d(
            in_channels=3,  # [h, r, t]
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        # Project concatenated feature maps to score
        self.fc = nn.Linear(n_filters * dim, 1)

    def forward(self, h, r, t):
        """
        Score triples (h, r, t).

        Args:
            h: Head entities (..., D)
            r: Relations (..., D)
            t: Tail entities (..., D)

        Returns:
            scores: Triple scores (...)
        """
        # Store original shape
        orig_shape = h.shape[:-1]

        # Flatten to batch dimension
        batch_size = h.reshape(-1, self.dim).shape[0]
        h_flat = h.reshape(batch_size, self.dim)
        r_flat = r.reshape(batch_size, self.dim)
        t_flat = t.reshape(batch_size, self.dim)

        # Stack [h, r, t]: (B, 3, k)
        stacked = torch.stack([h_flat, r_flat, t_flat], dim=1)  # (B, 3, k)

        # Apply 1D convolution: (B, n_filters, k)
        x = self.conv(stacked)
        x = F.relu(x)

        # Flatten: (B, n_filters * k)
        x = x.view(batch_size, -1)

        # Apply dropout
        x = self.dropout(x)

        # Project to score
        scores = self.fc(x).squeeze(-1)  # (B,)

        # Reshape back to original shape
        scores = scores.view(*orig_shape)

        return scores


class ConvEInteraction(nn.Module):
    """ConvE: Convolutional interaction matching reference implementation.

    Stacks head and relation embeddings vertically and applies 2D convolution.
    """

    def __init__(
        self,
        dim: int,
        emb_dim1: int,
        n_filters: int = 32,
        kernel_size: int = 3,
        input_drop: float = 0.2,
        hidden_drop: float = 0.3,
        feat_drop: float = 0.2,
        use_bias: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = dim // emb_dim1

        if dim % emb_dim1 != 0:
            raise ValueError(f"Embedding dim {dim} must be divisible by emb_dim1 {emb_dim1}")

        # Dropout layers
        self.inp_drop = nn.Dropout(input_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.feature_map_drop = nn.Dropout2d(feat_drop)

        # Convolutional layer
        self.conv1 = nn.Conv2d(1, n_filters, (kernel_size, kernel_size), stride=1, padding=0, bias=use_bias)

        conv_out_h = 2 * emb_dim1 - kernel_size + 1
        conv_out_w = self.emb_dim2 - kernel_size + 1
        hidden_size = n_filters * conv_out_h * conv_out_w

        # Fully connected layer to project back to embedding dimension
        self.fc = nn.Linear(hidden_size, dim)

    def forward(self, h, r, t):
        """
        Score triples (h, r, t).

        Args:
            h: Head entities (..., D)
            r: Relations (..., D)
            t: Tail entities (..., D)

        Returns:
            scores: Triple scores (...)
        """
        # Store original shape for later
        orig_shape = h.shape[:-1]

        # Flatten to batch dimension
        batch_size = h.reshape(-1, self.dim).shape[0]
        h_flat = h.reshape(batch_size, self.dim)
        r_flat = r.reshape(batch_size, self.dim)
        t_flat = t.reshape(batch_size, self.dim)

        # Reshape embeddings into 2D "images"
        h_img = h_flat.view(batch_size, 1, self.emb_dim1, self.emb_dim2)
        r_img = r_flat.view(batch_size, 1, self.emb_dim1, self.emb_dim2)

        # Stack head and relation vertically (along height dimension)
        stacked = torch.cat([h_img, r_img], dim=2)  # (B, 1, 2*emb_dim1, emb_dim2)

        # Input dropout
        x = self.inp_drop(stacked)

        # Convolution
        x = self.conv1(x)  # (B, n_filters, conv_out_h, conv_out_w)
        x = F.relu(x)

        # Feature map dropout
        x = self.feature_map_drop(x)

        # Flatten feature maps
        x = x.view(batch_size, -1)

        # Fully connected projection
        x = self.fc(x)  # (B, dim)
        x = self.hidden_drop(x)
        x = F.relu(x)

        # Score against tail entities (dot product)
        scores = (x * t_flat).sum(dim=-1)

        # Reshape back to original shape
        scores = scores.view(*orig_shape)

        return scores


class TriplesScoreLayer(nn.Module):
    """Wrapper for knowledge graph triple scoring interactions.

    Optimized for relation extraction in entity recognition models.

    Args:
        interaction_mode: The type of interaction to use. Available modes:
            - Translational: UM, SE, TransE, TransH, TransF, PairRE, TripleRE
            - Semantic: DistMult, SimplE, ComplEx, QuatE, HolE, DistMA
            - Neural: TuckER, ERMLP, ConvE, ConvKB
        dim: Embedding dimension (required for most interactions).
        **kwargs: Extra parameters for specific interactions:
            - TuckER: requires d_e, d_r, optional dropout
            - ERMLP: optional hidden (default 2048)
            - ConvE: requires emb_dim1, optional n_filters, kernel_size, input_drop,
            hidden_drop, feat_drop, use_bias
            - ConvKB: optional n_filters, dropout, use_bias
            - Norm-based (TransE, TransH, etc.): optional p, power, clamp_norm,
            use_scorer, dropout
    """

    # Define dimension requirements for each interaction
    DIMENSION_REQUIREMENTS = {
        "ComplEx": lambda d: d % 2 == 0,
        "SimplE": lambda d: d % 2 == 0,
        "QuatE": lambda d: d % 4 == 0,
    }

    def __init__(self, interaction_mode: str, dim: int = 768, **kwargs):
        super().__init__()
        self.mode = interaction_mode
        self.dim = dim

        # Validate dimension requirements
        self.validate_dimensions(dim)

        # Create the appropriate interaction
        if interaction_mode == "UM":
            self.interaction = UMInteraction(dim=dim, **kwargs)
        elif interaction_mode == "SE":
            self.interaction = SEInteraction(dim=dim, **kwargs)
        elif interaction_mode == "TransE":
            self.interaction = TransEInteraction(dim=dim, **kwargs)
        elif interaction_mode == "TransH":
            self.interaction = TransHInteraction(dim=dim, **kwargs)
        elif interaction_mode == "TransF":
            self.interaction = TransFInteraction(dim=dim, **kwargs)
        elif interaction_mode == "PairRE":
            self.interaction = PairREInteraction(dim=dim, **kwargs)
        elif interaction_mode == "TripleRE":
            self.interaction = TripleREInteraction(dim=dim, **kwargs)
        elif interaction_mode == "DistMult":
            self.interaction = DistMultInteraction()
        elif interaction_mode == "SimplE":
            self.interaction = SimplEInteraction(dim=dim)
        elif interaction_mode == "DistMA":
            self.interaction = DistMAInteraction()
        elif interaction_mode == "ComplEx":
            self.interaction = ComplExInteraction(dim=dim)
        elif interaction_mode == "QuatE":
            self.interaction = QuatEInteraction(dim=dim)
        elif interaction_mode == "HolE":
            self.interaction = HolEInteraction()
        elif interaction_mode == "TuckER":
            d_e = kwargs.get("d_e", dim)
            d_r = kwargs.get("d_r", dim)
            dropout = kwargs.get("dropout", 0.2)
            self.interaction = TuckERInteraction(d_e, d_r, dropout)
        elif interaction_mode == "ERMLP":
            hidden = kwargs.get("hidden", 2048)
            self.interaction = ERMLPInteraction(dim, hidden)
        elif interaction_mode == "ConvE":
            emb_dim1 = kwargs.get("emb_dim1")
            if emb_dim1 is None:
                raise ValueError("ConvE requires `emb_dim1` argument (height of reshaped embedding).")
            n_filters = kwargs.get("n_filters", 9)
            kernel_size = kwargs.get("kernel_size", 3)
            input_drop = kwargs.get("input_drop", 0.2)
            hidden_drop = kwargs.get("hidden_drop", 0.3)
            feat_drop = kwargs.get("feat_drop", 0.2)
            use_bias = kwargs.get("use_bias", True)
            self.interaction = ConvEInteraction(
                dim, emb_dim1, n_filters, kernel_size, input_drop, hidden_drop, feat_drop, use_bias
            )
        elif interaction_mode == "ConvKB":
            n_filters = kwargs.get("n_filters", 32)
            dropout = kwargs.get("dropout", 0.3)
            use_bias = kwargs.get("use_bias", True)
            self.interaction = ConvKBInteraction(dim, n_filters, dropout, use_bias)
        else:
            raise ValueError(f"Unknown interaction mode '{interaction_mode}'.")

    def validate_dimensions(self, dim: int):
        """
        Validate that the embedding dimension meets requirements for this interaction.

        Args:
            dim: The embedding dimension to validate

        Raises:
            ValueError: If dimension requirements are not met
        """
        if self.mode in self.DIMENSION_REQUIREMENTS:
            check = self.DIMENSION_REQUIREMENTS[self.mode]
            if not check(dim):
                if self.mode in ["ComplEx", "SimplE"]:
                    msg = f"{self.mode} requires even embedding dimension. Got {dim}."
                elif self.mode == "QuatE":
                    msg = f"{self.mode} requires embedding dimension divisible by 4. Got {dim}."
                else:
                    msg = f"{self.mode} has dimension requirements not satisfied by {dim}."
                raise ValueError(msg)

    def forward(self, h, r, t):
        """
        Score triples (h, r, t).

        Args:
            h: Head entities (..., D)
            r: Relations (..., D)
            t: Tail entities (..., D)

        Returns:
            scores: Triple scores (...)
        """
        return self.interaction(h, r, t)

    def forward_batched_relations(self, h, t, rel_embeddings):
        """
        Efficiently score entity pairs against all relation types.

        Args:
            h: Head entities (B, N, D)
            t: Tail entities (B, N, D)
            rel_embeddings: Relation type embeddings (B, C, D) or (C, D)

        Returns:
            scores: (B, N, C) scores for each pair against each relation type
        """
        B, N, D = h.shape

        # Handle both batched and unbatched relation embeddings
        if rel_embeddings.dim() == 2:
            C, _ = rel_embeddings.shape
            rel_embeddings = rel_embeddings.unsqueeze(0).expand(B, C, D)
        else:
            C = rel_embeddings.shape[1]

        # Expand dimensions for broadcasting
        h_exp = h.unsqueeze(2).expand(B, N, C, D)
        t_exp = t.unsqueeze(2).expand(B, N, C, D)
        r_exp = rel_embeddings.unsqueeze(1).expand(B, N, C, D)

        # Reshape to (B*N*C, D) for efficient batch processing
        h_flat = h_exp.reshape(B * N * C, D)
        r_flat = r_exp.reshape(B * N * C, D)
        t_flat = t_exp.reshape(B * N * C, D)

        # Score all triples at once
        scores_flat = self.interaction(h_flat, r_flat, t_flat)

        # Reshape back to (B, N, C)
        scores = scores_flat.view(B, N, C)

        return scores

    def forward_single_relation(self, h, t, r):
        """
        Score entity pairs with a single relation type.

        Args:
            h: Head entities (B, N, D)
            t: Tail entities (B, N, D)
            r: Single relation embedding (B, D) or (D,)

        Returns:
            scores: (B, N) scores for each pair with the given relation
        """
        B, N, D = h.shape

        # Expand relation to match batch and pair dimensions
        if r.dim() == 1:
            r = r.unsqueeze(0).unsqueeze(0).expand(B, N, D)
        elif r.dim() == 2:
            r = r.unsqueeze(1).expand(B, N, D)

        # Flatten for scoring
        h_flat = h.reshape(B * N, D)
        t_flat = t.reshape(B * N, D)
        r_flat = r.reshape(B * N, D)

        # Score
        scores_flat = self.interaction(h_flat, r_flat, t_flat)

        # Reshape to (B, N)
        scores = scores_flat.view(B, N)

        return scores
