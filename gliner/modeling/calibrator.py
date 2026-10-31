"""Calibration head for GLiNER: per-(span, label) inverse temperature and bias.

Adapted from the GLiClass calibrator. For every *candidate* pair, i.e. a (span, label)
whose raw probability is at least ``candidate_threshold``, an MLP predicts an inverse
temperature ``beta`` in ``(0, beta_max)`` and, optionally, a bias ``b``. The calibrated
logit is ``beta * z + b``. Pairs below the threshold keep their raw logit: they are
almost all trivial negatives, are never decoded at usual thresholds, and skipping them
keeps the head cheap (the full L x K x C grid is huge).

The head is fitted with the backbone frozen (see ``gliner/calibration.py``) and is
initialised to a global temperature / Platt fit, so it starts as global scaling.

Modes:
    * ``"span"``: logits (B, L, K, C) over the start x width grid, or (B, S, C)
      with explicit ``span_idx`` (B, S, 2). A pair is (span, label).
    * ``"token"``: logits (B, W, C, 3) with start / end / inside channels. A pair
      is (word, label); one beta / bias scales all three channels.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn

NUM_STAT_FEATURES = 6


def calibrator_mode(config: Any) -> str:
    """'token' for token-level models decoded from token logits, else 'span'."""
    if getattr(config, "span_mode", None) == "token_level" and not getattr(config, "represent_spans", False):
        return "token"
    return "span"


class GLiNERCalibrator(nn.Module):
    """MLP over (span repr, label repr, score statistics) -> (beta, bias) per candidate pair."""

    def __init__(
        self,
        word_dim: int,
        label_dim: int,
        mode: str = "span",
        proj_size: int = 128,
        hidden_size: int = 256,
        beta_max: float = 5.0,
        dropout: float = 0.1,
        use_bias: bool = True,
        candidate_threshold: float = 0.01,
        max_width: int = 12,
    ):
        super().__init__()
        if mode not in ("span", "token"):
            raise ValueError(f"mode must be 'span' or 'token', got {mode!r}")
        self.mode = mode
        self.beta_max = beta_max
        self.use_bias = use_bias
        self.candidate_threshold = candidate_threshold
        self.max_width = max_width

        # Span repr = [start word, end word, mean word]; projected first because there
        # can be many candidates per batch.
        self.word_proj = nn.Sequential(nn.LayerNorm(3 * word_dim), nn.Linear(3 * word_dim, proj_size))
        self.label_proj = nn.Sequential(nn.LayerNorm(label_dim), nn.Linear(label_dim, proj_size))
        self.mlp = nn.Sequential(
            nn.Linear(4 * proj_size + NUM_STAT_FEATURES, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 if use_bias else 1),
        )
        self.reset_to_global_scaling(1.0)

    @property
    def candidate_logit(self) -> float:
        p = min(max(self.candidate_threshold, 1e-12), 1 - 1e-12)
        return math.log(p / (1 - p))

    def reset_to_global_scaling(self, base_beta: float, base_bias: float = 0.0) -> None:
        """Zero the output layer so (beta, b) == (base_beta, base_bias) for every pair."""
        ratio = min(max(base_beta / self.beta_max, 1e-4), 1 - 1e-4)
        output = self.mlp[-1]
        with torch.no_grad():
            output.weight.zero_()
            output.bias[0] = math.log(ratio / (1 - ratio))
            if self.use_bias:
                output.bias[1] = base_bias

    # ------------------------------------------------------------------
    # Candidate selection and features (no parameters involved)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gather(
        self,
        logits: torch.Tensor,
        words_embedding: torch.Tensor,
        prompts_embedding: torch.Tensor,
        word_mask: torch.Tensor,
        class_mask: torch.Tensor,
        span_idx: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Select candidate pairs and build their inputs.

        Returns a dict with ``index`` (tuple of index tensors into ``logits``), ``z``
        (n,) or (n, 3) raw logits, ``word_feats`` (n, 3 * Dw), ``label_feats`` (n, Dl)
        and ``stats`` (n, NUM_STAT_FEATURES).
        """
        z = logits.float()
        C = min(z.shape[-2] if self.mode == "token" else z.shape[-1], prompts_embedding.shape[1], class_mask.shape[1])
        if self.mode == "token":
            z = z[:, :, :C]
        else:
            z = z[..., :C]
        class_valid = class_mask[:, :C].bool()
        num_words = word_mask.bool().sum(-1)  # (B,)
        num_labels = class_valid.sum(-1).clamp(min=1).float()  # (B,)

        if self.mode == "token":
            return self._gather_token(z, words_embedding, prompts_embedding, word_mask, class_valid, num_labels)

        if z.dim() == 4:  # (B, L, K, C) grid
            B, L, K, _ = z.shape
            starts = torch.arange(L, device=z.device).view(1, L, 1).expand(B, L, K)
            ends = starts + torch.arange(K, device=z.device).view(1, 1, K)
            span_valid = ends < num_words.view(B, 1, 1)
        else:  # (B, S, C) explicit spans
            if span_idx is None:
                raise ValueError("span logits of shape (B, S, C) need span_idx")
            starts, ends = span_idx[..., 0], span_idx[..., 1]
            span_valid = (ends < num_words.view(-1, 1)) & (starts >= 0) & (starts <= ends)
            if span_mask is not None:
                span_valid = span_valid & span_mask.bool()

        masked = z.masked_fill(~class_valid.view(class_valid.size(0), *([1] * (z.dim() - 2)), -1), float("-inf"))
        cand = (masked >= self.candidate_logit) & span_valid.unsqueeze(-1)
        index = cand.nonzero(as_tuple=True)
        b, c = index[0], index[-1]
        pos = index[1:-1]  # (l, k) or (s,)
        start, end = starts[(b, *pos)], ends[(b, *pos)]

        zc = z[index]
        best = masked.amax(-1)[(b, *pos)]
        mean = (z * class_valid.view(class_valid.size(0), *([1] * (z.dim() - 2)), -1)).sum(-1)[(b, *pos)]
        mean = mean / num_labels[b]
        width = (end - start + 1).float() / self.max_width
        stats = torch.stack([zc, zc.abs(), zc - best, zc - mean, width, num_labels[b].log()], dim=-1)

        word_feats = self._span_word_features(words_embedding, b, start, end)
        return {
            "index": index,
            "z": zc,
            "word_feats": word_feats,
            "label_feats": prompts_embedding[b, c].float(),
            "stats": stats,
        }

    def _gather_token(self, z, words_embedding, prompts_embedding, word_mask, class_valid, num_labels):
        W = min(z.shape[1], words_embedding.shape[1], word_mask.shape[1])
        z = z[:, :W]
        word_valid = word_mask[:, :W].bool()
        valid = word_valid[:, :, None] & class_valid[:, None, :]
        cand = (z.amax(-1) >= self.candidate_logit) & valid
        index = cand.nonzero(as_tuple=True)  # (b, w, c)
        b, w, c = index

        zc = z[index]  # (n, 3)
        inside = z[..., 2].masked_fill(~class_valid[:, None, :], float("-inf"))
        best = inside.amax(-1)[b, w]
        mean = (z[..., 2] * class_valid[:, None, :]).sum(-1)[b, w] / num_labels[b]
        zi = zc[:, 2]
        stats = torch.stack([zc[:, 0], zc[:, 1], zi, zi - best, zi - mean, num_labels[b].log()], dim=-1)

        word = words_embedding[b, w].float()
        return {
            "index": index,
            "z": zc,
            "word_feats": torch.cat([word, word, word], dim=-1),
            "label_feats": prompts_embedding[b, c].float(),
            "stats": stats,
        }

    @staticmethod
    def _span_word_features(words_embedding, b, start, end):
        words = words_embedding.float()
        prefix = torch.nn.functional.pad(words.cumsum(1), (0, 0, 1, 0))  # (B, W + 1, D)
        mean = (prefix[b, end + 1] - prefix[b, start]) / (end - start + 1).unsqueeze(-1).float()
        return torch.cat([words[b, start], words[b, end], mean], dim=-1)

    # ------------------------------------------------------------------
    # Head
    # ------------------------------------------------------------------

    def predict(self, word_feats, label_feats, stats):
        """(beta, bias) of shape (n,) for gathered candidates."""
        dtype = self.mlp[0].weight.dtype
        w = self.word_proj(word_feats.to(dtype))
        lab = self.label_proj(label_feats.to(dtype))
        features = torch.cat([w, lab, w * lab, (w - lab).abs(), stats.to(dtype)], dim=-1)
        out = self.mlp(features).float()
        # Clamped away from 0 so a positive logit never becomes exactly 0 (sign is preserved).
        beta = (self.beta_max * torch.sigmoid(out[:, 0])).clamp(min=1e-6)
        bias = out[:, 1] if self.use_bias else torch.zeros_like(beta)
        return beta, bias

    def forward(self, logits, words_embedding, prompts_embedding, word_mask, class_mask, span_idx=None, span_mask=None):
        """Gather candidates and predict their (beta, bias). Returns the gathered dict plus 'beta', 'bias'."""
        gathered = self.gather(logits, words_embedding, prompts_embedding, word_mask, class_mask, span_idx, span_mask)
        if gathered["z"].shape[0] == 0:
            empty = gathered["z"].new_zeros(0)
            return {**gathered, "beta": empty, "bias": empty}
        beta, bias = self.predict(gathered["word_feats"], gathered["label_feats"], gathered["stats"])
        return {**gathered, "beta": beta, "bias": bias}

    @staticmethod
    def calibrated_candidate_logits(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calibrated logits ``beta * z + b`` of gathered candidates, (n,) or (n, 3)."""
        beta, bias = out["beta"], out["bias"]
        if out["z"].dim() == 2:
            beta, bias = beta.unsqueeze(-1), bias.unsqueeze(-1)
        return beta * out["z"] + bias

    @torch.no_grad()
    def calibrate(
        self, logits, words_embedding, prompts_embedding, word_mask, class_mask, span_idx=None, span_mask=None
    ):
        """Return a copy of ``logits`` with candidate pairs replaced by calibrated logits."""
        out = self.forward(logits, words_embedding, prompts_embedding, word_mask, class_mask, span_idx, span_mask)
        calibrated = logits.clone()
        if out["z"].shape[0]:
            calibrated[out["index"]] = self.calibrated_candidate_logits(out).to(logits.dtype)
        return calibrated


def build_calibrator(config: Any) -> GLiNERCalibrator:
    """Build a calibrator from ``calibrator_*`` config fields."""
    return GLiNERCalibrator(
        word_dim=config.calibrator_word_dim or config.hidden_size,
        label_dim=config.calibrator_label_dim or config.hidden_size,
        mode=config.calibrator_mode or calibrator_mode(config),
        proj_size=config.calibrator_proj_size,
        hidden_size=config.calibrator_hidden_size,
        beta_max=config.calibrator_beta_max,
        dropout=config.calibrator_dropout,
        use_bias=config.calibrator_use_bias,
        candidate_threshold=config.calibrator_candidate_threshold,
        max_width=getattr(config, "max_width", 12) or 12,
    )


def apply_calibrator(calibrator: GLiNERCalibrator, output: Any) -> Any:
    """Calibrate the logits that get decoded, keeping the raw ones in ``uncalibrated_*``."""
    words = getattr(output, "words_embedding", None)
    prompts = getattr(output, "prompts_embedding", None)
    word_mask = getattr(output, "mask", None)
    class_mask = getattr(output, "prompts_embedding_mask", None)
    if words is None or prompts is None or word_mask is None or class_mask is None:
        return output

    span_logits = getattr(output, "span_logits", None)
    if span_logits is not None and getattr(output, "span_idx", None) is not None and calibrator.mode == "span":
        # Token models with represent_spans decode from span_logits.
        output.uncalibrated_span_logits = span_logits
        output.span_logits = calibrator.calibrate(
            span_logits, words, prompts, word_mask, class_mask, output.span_idx, output.span_mask
        )
        return output

    logits = output.logits
    if logits is None:
        return output
    span_idx = output.span_idx if logits.dim() == 3 else None
    span_mask = output.span_mask if logits.dim() == 3 else None
    output.uncalibrated_logits = logits
    output.logits = calibrator.calibrate(logits, words, prompts, word_mask, class_mask, span_idx, span_mask)
    return output
