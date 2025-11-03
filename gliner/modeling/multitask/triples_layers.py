import math
import torch
import torch.nn as nn
import torch.fft as fft
from torch.nn import functional as F


def _split_complex(x):
    """Assume last dim = 2k and split into real / imag parts."""
    return torch.chunk(x, 2, dim=-1)

def _split_quaternion(x):
    """Assume last dim = 4k and split into (1,i,j,k) parts."""
    return torch.chunk(x, 4, dim=-1)

def _norm_clamp(x, max_norm):
    return x.clamp(max=max_norm) if max_norm is not None else x

class NormBasedInteraction(nn.Module):
    def __init__(self, p: int = 2, power: float = 1., clamp_norm: float | None = None):
        """
        p ........... ℓ_p norm (e.g. 1 or 2) used in ‖·‖_p
        power ....... raise norm to this power (TransE² style) before negating
        clamp_norm .. optional upper bound for numerical stability
        """
        super().__init__()
        self.p, self.power, self.clamp = p, power, clamp_norm

    def _score(self, x):
        """
        x ...... (..., D) residual vector
        returns . negative distance (larger = better)
        """
        d = torch.linalg.norm(x, ord=self.p, dim=-1)
        d = d.pow(self.power)
        d = _norm_clamp(d, self.clamp)
        return -d        # KGE convention: higher score ⇒ more plausible

class UMInteraction(NormBasedInteraction):
    """Unstructured model ‖h - t‖."""
    def forward(self, h, r, t):
        return self._score(h - t)

class TransEInteraction(NormBasedInteraction):
    """TransE ‖h + r − t‖."""
    def forward(self, h, r, t):
        return self._score(h + r - t)

class SEInteraction(NormBasedInteraction):
    """
    Structure Embedding (SE)
    Uses head / tail specific diagonal matrices built from relation.
    h' = diag(r) · h    ,    t' = diag(r) · t
    """
    def forward(self, h, r, t):
        diag = torch.diag_embed(r)                 # (..., D, D)
        h_ = torch.matmul(diag, h.unsqueeze(-1)).squeeze(-1)
        t_ = torch.matmul(diag, t.unsqueeze(-1)).squeeze(-1)
        return self._score(h_ - t_)

class TransRInteraction(NormBasedInteraction):
    """
    TransR with per-sample projection matrix M_r = r · rᵀ / ‖r‖ (rank-1)
    h' = M_r h , t' = M_r t
    """
    def forward(self, h, r, t):
        r_unit = F.normalize(r, dim=-1)                    # (..., D)
        M = r_unit.unsqueeze(-1) * r_unit.unsqueeze(-2)    # outer product
        h_, t_ = (torch.matmul(M, x.unsqueeze(-1)).squeeze(-1) for x in (h, t))
        return self._score(h_ + r - t_)

class TransDInteraction(NormBasedInteraction):
    """
    TransD – low-rank projections with extra “projection” vectors.
    We split the relation into (r_vec , p_head , p_tail) each of dim D.
    """
    def forward(self, h, r, t):
        r_vec, p_h, p_t = torch.chunk(r, 3, dim=-1)
        #   M_h = I + p_h ⊗ r_vec ,   M_t = I + p_t ⊗ r_vec
        M_h = torch.eye(h.shape[-1], device=h.device) + p_h.unsqueeze(-1) * r_vec.unsqueeze(-2)
        M_t = torch.eye(t.shape[-1], device=t.device) + p_t.unsqueeze(-1) * r_vec.unsqueeze(-2)
        h_ = torch.matmul(M_h, h.unsqueeze(-1)).squeeze(-1)
        t_ = torch.matmul(M_t, t.unsqueeze(-1)).squeeze(-1)
        return self._score(h_ + r_vec - t_)

class TransHInteraction(NormBasedInteraction):
    """
    TransH – project to relation hyper-plane with normal w.
    Relation vector r is split into (r_tr , w). w is ℓ₂-normalised.
    """
    def forward(self, h, r, t):
        r_tr, w = torch.chunk(r, 2, dim=-1)
        w = F.normalize(w, dim=-1)
        def proj(x): return x - (x * w).sum(dim=-1, keepdim=True) * w
        return self._score(proj(h) + r_tr - proj(t))

class PairREInteraction(NormBasedInteraction):
    """
    PairRE – per-relation element-wise scaling of h & t, split r→(α,β)
    score = ‖α∘h − β∘t‖
    """
    def forward(self, h, r, t):
        alpha, beta = torch.chunk(r, 2, dim=-1)
        return self._score(alpha * h - beta * t)

class LineaREInteraction(NormBasedInteraction):
    """
    LineaRE – PairRE + translation δ
    r split into (α,β,δ)
    """
    def forward(self, h, r, t):
        alpha, beta, delta = torch.chunk(r, 3, dim=-1)
        return self._score(alpha * h + delta - beta * t)

class TripleREInteraction(NormBasedInteraction):
    """
    TripleRE – LineaRE + global scaling γ (scalar per relation).
    r split into (α,β,δ,γ) ; γ is last single feature (broadcast).
    """
    def forward(self, h, r, t):
        *abc, gamma = torch.split(r, r.shape[-1]-1, dim=-1)
        alpha, beta, delta = torch.chunk(abc[0], 3, dim=-1)
        return gamma.squeeze(-1) * self._score(alpha * h + delta - beta * t)

class RotatEInteraction(NormBasedInteraction):
    """
    RotatE – treat r as phase vector ϕ, h/t as complex.
    Requires last dim = 2k (real, imag).
    """
    def __init__(self, clamp_norm=None):
        super().__init__(p=2, power=1., clamp_norm=clamp_norm)

    def forward(self, h, r, t):
        h_re, h_im = _split_complex(h)
        t_re, t_im = _split_complex(t)
        phase = r / math.pi                          # keep small magnitude
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        # rotate
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re
        diff = torch.cat([rot_re - t_re, rot_im - t_im], dim=-1)
        return self._score(diff)

class DistMultInteraction(nn.Module):
    """DistMult – Σ_d h_d r_d t_d."""
    def forward(self, h, r, t):
        return (h * r * t).sum(dim=-1)

class CPInteraction(nn.Module):
    """Canonical Polyadic (CP/KANDE).  Same formula, but embeddings come from different lookup tables."""
    def forward(self, h, r, t):
        return (h * r * t).sum(dim=-1)

class SimplEInteraction(nn.Module):
    """
    SimplE – split every embedding into (forward, backward) halves.
    score = ½( ⟨h_f , r_f , t_b⟩ + ⟨t_f , r_b , h_b⟩ )
    """
    def forward(self, h, r, t):
        h_f, h_b = _split_complex(h)            # split into two
        t_f, t_b = _split_complex(t)
        r_f, r_b = _split_complex(r)
        s1 = (h_f * r_f * t_b).sum(dim=-1)
        s2 = (t_f * r_b * h_b).sum(dim=-1)
        return 0.5 * (s1 + s2)

class RESCALInteraction(nn.Module):
    """
    RESCAL – relation-specific full matrix W_r (flattened into r, shape D×D).
    r has last dim = D².
    """
    def forward(self, h, r, t):
        D = h.shape[-1]
        W = r.view(*r.shape[:-1], D, D)
        ht = torch.matmul(h.unsqueeze(-2), W).squeeze(-2)  # (..., D)
        return (ht * t).sum(dim=-1)

class TuckERInteraction(nn.Module):
    """
    TuckER – global core tensor W (D_e × D_r × D_e).
    """
    def __init__(self, d_e: int, d_r: int, dropout: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_r, d_e, d_e))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(d_e)
        self.bn1 = nn.BatchNorm1d(d_e)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, r, t):
        # batchnorm on entities (reshape to 2-D first)
        h = self.bn0(h.view(-1, h.shape[-1])).view_as(h)
        t = self.bn1(t.view(-1, t.shape[-1])).view_as(t)

        # core interaction
        W_mat = torch.matmul(r, self.W)      # (..., D_e, D_e)
        hr = torch.matmul(h.unsqueeze(-2), W_mat).squeeze(-2)
        scores = (hr * t).sum(dim=-1)
        return scores

class DistMAInteraction(nn.Module):
    """DistMA – sum of pairwise dot products."""
    def forward(self, h, r, t):
        return (h * r).sum(dim=-1) + (h * t).sum(dim=-1) + (r * t).sum(dim=-1)

class TransFInteraction(NormBasedInteraction):
    """
    TransF – element-wise relation-specific scaling before translation.
    r split into (r_vec , α , β)
    """
    def forward(self, h, r, t):
        r_vec, alpha, beta = torch.chunk(r, 3, dim=-1)
        h_ = alpha * h
        t_ = beta * t
        return self._score(h_ + r_vec - t_)

class ComplExInteraction(nn.Module):
    """
    ComplEx – Re( ⟨h , r , conj(t)⟩ ) with complex embeddings.
    """
    def forward(self, h, r, t):
        h_re, h_im = _split_complex(h)
        r_re, r_im = _split_complex(r)
        t_re, t_im = _split_complex(t)
        return (h_re * r_re * t_re
              + h_re * r_im * t_im
              + h_im * r_re * t_im
              - h_im * r_im * t_re).sum(dim=-1)

class QuatEInteraction(nn.Module):
    """QuatE – use Hamilton product (a,b,c,d)⨂(e,f,g,h)."""
    def forward(self, h, r, t):
        h0, h1, h2, h3 = _split_quaternion(h)
        r0, r1, r2, r3 = _split_quaternion(r)
        t0, t1, t2, t3 = _split_quaternion(t)
        # Hamilton product h ⨂ r
        A0 =  h0*r0 - h1*r1 - h2*r2 - h3*r3
        A1 =  h0*r1 + h1*r0 + h2*r3 - h3*r2
        A2 =  h0*r2 - h1*r3 + h2*r0 + h3*r1
        A3 =  h0*r3 + h1*r2 - h2*r1 + h3*r0
        return (A0 * t0 + A1 * t1 + A2 * t2 + A3 * t3).sum(dim=-1)

class HolEInteraction(nn.Module):
    """HolE – circular correlation ϕ(h, t) · r."""
    def forward(self, h, r, t):
        # FFT-based circular correlation
        fft_h = fft.rfft(h, dim=-1)
        fft_t = fft.rfft(t, dim=-1)
        corr = fft.irfft(fft_h.conj() * fft_t, n=h.shape[-1], dim=-1)
        return (corr * r).sum(dim=-1)

class AutoSFInteraction(nn.Module):
    """
    Very simplified AutoSF: score = Σ_i a_i ⟨block_i(h,r), block_i(t)⟩
    We use a learnable scalar per block.
    """
    def __init__(self, n_blocks: int, dim: int):
        super().__init__()
        self.scales = nn.Parameter(torch.ones(n_blocks))
        self.n_blocks = n_blocks
        self.dim = dim // n_blocks

    def forward(self, h, r, t):
        h_blocks = h.split(self.dim, dim=-1)
        r_blocks = r.split(self.dim, dim=-1)
        t_blocks = t.split(self.dim, dim=-1)
        scores = []
        for i, (hb, rb, tb) in enumerate(zip(h_blocks, r_blocks, t_blocks)):
            scores.append(self.scales[i] * (hb * rb * tb).sum(dim=-1))
        return torch.stack(scores, dim=0).sum(dim=0)
    

class ERMLPInteraction(nn.Module):
    """ER-MLP 2-layer perceptron on concat."""
    def __init__(self, dim: int, hidden: int = 2_048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h, r, t):
        x = torch.cat([h, r, t], dim=-1)
        return self.mlp(x).squeeze(-1)

class ConvEInteraction(nn.Module):
    """
    Minimal ConvE (shares filter for all relations, no BN/dropout for brevity).
    """
    def __init__(self, dim: int, height: int, width: int, n_filters: int = 32, k: int = 3):
        super().__init__()
        self.h, self.w = height, width
        self.conv = nn.Conv2d(1, n_filters, kernel_size=k, padding=k//2)
        flat = n_filters * height * width
        self.fc = nn.Linear(flat, dim)

    def forward(self, h, r, t):
        # reshape to "image": (B, 1, H, 2*W)  (stack h & r)
        B = h.shape[0]
        img = torch.cat([h, r], dim=-1).view(B, 1, self.h, 2*self.w)
        act = F.relu(self.conv(img))
        act = act.view(B, -1)
        out = F.relu(self.fc(act))
        return (out * t).sum(dim=-1)


INTERACTIONS = {
    "UM": UMInteraction(),
    "TransE": TransEInteraction(p=1),
    "SE": SEInteraction(),
    "TransR": TransRInteraction(),
    "TransD": TransDInteraction(),
    "TransH": TransHInteraction(),
    "PairRE": PairREInteraction(),
    "LineaRE": LineaREInteraction(),
    "TripleRE": TripleREInteraction(),
    "RotatE": RotatEInteraction(),
    "DistMult": DistMultInteraction(),
    "CP": CPInteraction(),
    "SimplE": SimplEInteraction(),
    "RESCAL": RESCALInteraction(),
    # TuckER needs dims ⇒ build separately
    "DistMA": DistMAInteraction(),
    "TransF": TransFInteraction(),
    "ComplEx": ComplExInteraction(),
    "QuatE": QuatEInteraction(),
    "HolE": HolEInteraction(),
    # AutoSF & neural: create as needed
}


class TriplesScoreLayer(nn.Module):
    """
    Wrapper for knowledge graph triple scoring interactions.
    Allows selecting any interaction mode (e.g., TransE, RotatE, ComplEx, etc.)
    and applies it to (h, r, t).

    Parameters
    ----------
    interaction_mode : str
        The type of interaction to use. Must be one of:
        'UM', 'TransE', 'SE', 'TransR', 'TransD', 'TransH',
        'PairRE', 'LineaRE', 'TripleRE', 'RotatE', 'DistMult',
        'CP', 'SimplE', 'RESCAL', 'DistMA', 'TransF',
        'ComplEx', 'QuatE', 'HolE', 'AutoSF', 'ERMLP', 'ConvE'
    **kwargs : dict
        Extra parameters for interactions that need them (e.g., TuckER dims, ConvE image size, etc.).
    """

    def __init__(self, interaction_mode: str, **kwargs):
        super().__init__()
        mode = interaction_mode

        if mode in INTERACTIONS:
            self.interaction = INTERACTIONS[mode]

        elif mode == "TuckER":
            d_e = kwargs.get("d_e")
            d_r = kwargs.get("d_r")
            if d_e is None or d_r is None:
                raise ValueError("TuckER requires `d_e` and `d_r` arguments.")
            dropout = kwargs.get("dropout", 0.2)
            self.interaction = TuckERInteraction(d_e, d_r, dropout)

        elif mode == "AutoSF":
            n_blocks = kwargs.get("n_blocks")
            dim = kwargs.get("dim")
            if n_blocks is None or dim is None:
                raise ValueError("AutoSF requires `n_blocks` and `dim` arguments.")
            self.interaction = AutoSFInteraction(n_blocks, dim)

        elif mode == "ERMLP":
            dim = kwargs.get("dim")
            hidden = kwargs.get("hidden", 2048)
            if dim is None:
                raise ValueError("ERMLP requires `dim` argument.")
            self.interaction = ERMLPInteraction(dim, hidden)

        elif mode == "ConvE":
            dim = kwargs.get("dim")
            height = kwargs.get("height")
            width = kwargs.get("width")
            n_filters = kwargs.get("n_filters", 32)
            k = kwargs.get("k", 3)
            if dim is None or height is None or width is None:
                raise ValueError("ConvE requires `dim`, `height`, and `width` arguments.")
            self.interaction = ConvEInteraction(dim, height, width, n_filters, k)

        else:
            raise ValueError(f"Unknown interaction mode '{interaction_mode}'.")

    def forward(self, h, r, t):
        return self.interaction(h, r, t)