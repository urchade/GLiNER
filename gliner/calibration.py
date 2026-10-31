"""Post-hoc probability calibration for GLiNER.

A trained model's logits ``s`` are mapped to probabilities with

    p = sigmoid(s / T + b)

where the temperature ``T`` and bias ``b`` live on the model config
(``logit_temperature`` / ``logit_bias`` for entities and
``rel_logit_temperature`` / ``rel_logit_bias`` for relations). The defaults
``T=1, b=0`` leave predictions unchanged. Every decoder goes through
:func:`calibrated_sigmoid`, so a calibrated config changes the scores (and,
with ``b != 0``, which spans pass the threshold) without touching the weights.

:func:`fit_logit_calibration` fits ``T`` (and optionally ``b``) on held-out
logits while the model stays frozen; ``scripts/calibrate.py`` runs it end to end.
"""

from __future__ import annotations

import math
import random
import logging
import numbers
from typing import Any, Dict, List, Tuple, Iterable, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CALIBRATION_ATTRS = {
    "entity": ("logit_temperature", "logit_bias"),
    "relation": ("rel_logit_temperature", "rel_logit_bias"),
}


def get_logit_calibration(config: Any, kind: str = "entity") -> Tuple[float, float]:
    """Return ``(temperature, bias)`` stored on ``config`` for ``kind`` ('entity' or 'relation')."""
    temp_attr, bias_attr = _CALIBRATION_ATTRS[kind]
    temperature = getattr(config, temp_attr, None)
    bias = getattr(config, bias_attr, None)
    # Missing or non-numeric values (old configs, mocks) mean "uncalibrated".
    temperature = float(temperature) if isinstance(temperature, numbers.Real) else 1.0
    bias = float(bias) if isinstance(bias, numbers.Real) else 0.0
    if temperature <= 0:
        raise ValueError(f"{temp_attr} must be positive, got {temperature}")
    return temperature, bias


def apply_logit_calibration(logits: torch.Tensor, config: Any, kind: str = "entity") -> torch.Tensor:
    """Return ``logits / T + b``; returns ``logits`` itself when uncalibrated."""
    temperature, bias = get_logit_calibration(config, kind)
    if temperature == 1.0 and bias == 0.0:
        return logits
    return logits / temperature + bias


def calibrated_sigmoid(logits: torch.Tensor, config: Any, kind: str = "entity") -> torch.Tensor:
    """Sigmoid of calibrated logits. Drop-in replacement for ``torch.sigmoid`` in decoders."""
    return torch.sigmoid(apply_logit_calibration(logits, config, kind))


def set_logit_calibration(config: Any, temperature: float, bias: float = 0.0, kind: str = "entity") -> None:
    """Store ``(temperature, bias)`` on ``config`` for ``kind``."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    temp_attr, bias_attr = _CALIBRATION_ATTRS[kind]
    setattr(config, temp_attr, float(temperature))
    setattr(config, bias_attr, float(bias))


# ---------------------------------------------------------------------------
# Fitting and metrics
# ---------------------------------------------------------------------------


def fit_logit_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    fit_bias: bool = False,
    max_iter: int = 200,
    device: str | torch.device | None = None,
) -> Tuple[float, float]:
    """Fit temperature (and optionally bias) by minimising weighted BCE on frozen logits.

    Args:
        logits: 1-D tensor of raw logits for (row, class) entries.
        labels: 1-D tensor of {0, 1} targets, same length.
        weights: Optional 1-D importance weights (e.g. ``1 / keep_prob`` for
            subsampled negatives).
        fit_bias: Also fit an additive bias ``b``. Changes which entries
            pass a fixed threshold; with ``False`` only confidence changes.
        max_iter: L-BFGS iterations.
        device: Device for the optimisation. Defaults to ``logits.device``.

    Returns:
        ``(temperature, bias)``.
    """
    device = device or logits.device
    x = logits.detach().to(device=device, dtype=torch.float64)
    y = labels.detach().to(device=device, dtype=torch.float64)
    w = None if weights is None else weights.detach().to(device=device, dtype=torch.float64)

    # Parametrise log T so the temperature stays positive.
    log_t = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, device=device, requires_grad=fit_bias)
    params = [log_t, bias] if fit_bias else [log_t]
    optimizer = torch.optim.LBFGS(params, lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = _weighted_bce(x / log_t.exp() + bias, y, w)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item()), float(bias.item())


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if weights is None:
        return loss.mean()
    return (loss * weights).sum() / weights.sum()


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    temperature: float = 1.0,
    bias: float = 0.0,
    n_bins: int = 15,
    threshold: float = 0.5,
    min_confidence: float = 0.1,
) -> Dict[str, float]:
    """Calibration and thresholded quality metrics over (row, class) entries.

    Returns NLL, Brier score, expected calibration error (ECE) over all
    entries, ECE restricted to entries with ``p >= min_confidence`` (the
    region where thresholds act; the full ECE is dominated by easy
    negatives), and entry-level precision / recall / F1 at ``threshold``.
    """
    x = logits.detach().double()
    y = labels.detach().double()
    w = torch.ones_like(x) if weights is None else weights.detach().double()

    z = x / temperature + bias
    p = torch.sigmoid(z)

    total = w.sum()
    nll = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / total
    brier = (((p - y) ** 2) * w).sum() / total

    pred = p > threshold
    tp = (w * (pred & (y > 0.5))).sum()
    fp = (w * (pred & (y <= 0.5))).sum()
    fn = (w * (~pred & (y > 0.5))).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0)
    recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

    confident = p >= min_confidence
    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": _ece(p, y, w, n_bins),
        "ece_confident": _ece(p[confident], y[confident], w[confident], n_bins),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_entries": float(total),
        "num_positives": float((w * y).sum()),
    }


def _ece(p: torch.Tensor, y: torch.Tensor, w: torch.Tensor, n_bins: int) -> float:
    total = w.sum()
    if p.numel() == 0 or total <= 0:
        return float("nan")
    bins = torch.clamp((p * n_bins).long(), max=n_bins - 1)
    ece = torch.zeros((), dtype=torch.float64)
    for b in range(n_bins):
        in_bin = bins == b
        bw = w[in_bin].sum()
        if bw <= 0:
            continue
        conf = (p[in_bin] * w[in_bin]).sum() / bw
        acc = (y[in_bin] * w[in_bin]).sum() / bw
        ece += bw / total * (conf - acc).abs()
    return float(ece)


# ---------------------------------------------------------------------------
# Logit collection from a frozen GLiNER model
# ---------------------------------------------------------------------------


def dataset_entity_types(data: Iterable[Dict[str, Any]], key: str = "ner") -> List[str]:
    """Sorted union of entity labels in a GLiNER-format dataset."""
    labels = set()
    for example in data:
        labels.update(str(ent[-1]) for ent in example.get(key, []))
        labels.update(example.get(f"{key}_labels", []))
        labels.update(example.get(f"{key}_negatives", []))
    return sorted(labels)


def freeze_model(model: torch.nn.Module) -> None:
    """Put ``model`` in eval mode and disable gradients for all its parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def _entity_entries(output: Any, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten (logits, labels) of valid (row, class) entries from one forward pass."""
    logits = output.logits
    labels = batch["labels"]
    class_mask = output.prompts_embedding_mask.bool()  # (B, C)
    B = logits.size(0)

    if logits.dim() == 4 and labels.dim() == 4:
        # Token-level: (B, W, C, 3) start / end / inside.
        W = min(logits.size(1), labels.size(1))
        C = min(logits.size(2), labels.size(2), class_mask.size(1))
        logits, labels = logits[:, :W, :C], labels[:, :W, :C]
        word_mask = output.mask[:, :W].bool()
        valid = word_mask[:, :, None, None] & class_mask[:, None, :C, None]
        valid = valid.expand_as(logits)
    else:
        # Span-level: logits (B, L, K, C) or (B, S, C); labels (B, S, C).
        logits = logits.reshape(B, -1, logits.size(-1))
        S = min(logits.size(1), labels.size(1))
        C = min(logits.size(2), labels.size(2), class_mask.size(1))
        logits, labels = logits[:, :S, :C], labels[:, :S, :C]
        span_mask = batch.get("span_mask")
        if span_mask is None:
            span_mask = torch.ones(B, S, dtype=torch.bool, device=logits.device)
        span_mask = span_mask.reshape(B, -1)[:, :S].bool()
        valid = span_mask[:, :, None] & class_mask[:, None, :C]

    valid = valid & (labels >= 0) & torch.isfinite(logits)
    return logits[valid].float(), labels[valid].float()


@torch.inference_mode()
def collect_entity_logits(
    model: Any,
    data: Sequence[Dict[str, Any]],
    batch_size: int = 8,
    entity_types: Sequence[str] | None = None,
    labels_per_pass: int | None = None,
    neg_keep_prob: float = 1.0,
    seed: int = 42,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a frozen GLiNER model over a labelled dataset and gather entity logits.

    Args:
        model: A torch-runtime GLiNER model (``GLiNER.from_pretrained(...)``).
        data: GLiNER-format examples: ``{"tokenized_text": [...], "ner": [[start, end, label], ...]}``.
        batch_size: Examples per forward pass.
        entity_types: Fixed label set given to every example (mirrors
            inference). ``None`` uses training-style per-example labels
            (gold types plus sampled in-batch negatives).
        labels_per_pass: Split ``entity_types`` into chunks of this size,
            one forward pass per chunk. ``None`` uses all labels at once.
        neg_keep_prob: Keep each negative entry with this probability (to save
            memory); kept negatives get weight ``1 / neg_keep_prob``.
        seed: Seed for negative subsampling and label sampling.
        show_progress: Log progress at INFO level.

    Returns:
        ``(logits, labels, weights)``: 1-D CPU tensors over valid entries.
    """
    if getattr(model, "is_runtime_model", False):
        raise ValueError("Calibration needs a torch-runtime model; load it without ONNX/OpenVINO.")
    if not 0.0 < neg_keep_prob <= 1.0:
        raise ValueError("neg_keep_prob must be in (0, 1]")

    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)
    collator = model._create_data_collator()
    device = model.device

    label_chunks: List[List[str] | None]
    if entity_types is None:
        label_chunks = [None]
    else:
        entity_types = list(entity_types)
        step = labels_per_pass or len(entity_types) or 1
        label_chunks = [entity_types[i : i + step] for i in range(0, len(entity_types), step)]

    all_logits, all_labels, all_weights = [], [], []
    num_batches = math.ceil(len(data) / batch_size) * len(label_chunks)
    done = 0
    for chunk in label_chunks:
        for start in range(0, len(data), batch_size):
            examples = list(data[start : start + batch_size])
            if chunk is None:
                batch = collator(examples)
            else:
                batch = collator(examples, entity_types=[list(chunk) for _ in examples])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            output = model(**batch)
            logits, labels = _entity_entries(output, batch)
            logits, labels = logits.cpu(), labels.cpu()

            weights = torch.ones_like(logits)
            if neg_keep_prob < 1.0:
                is_neg = labels < 0.5
                keep = ~is_neg | (torch.rand(logits.shape, generator=generator) < neg_keep_prob)
                logits, labels = logits[keep], labels[keep]
                weights = torch.where(is_neg[keep], 1.0 / neg_keep_prob, 1.0)

            all_logits.append(logits)
            all_labels.append(labels)
            all_weights.append(weights)

            done += 1
            if show_progress and (done % 10 == 0 or done == num_batches):
                logger.info("collected %d/%d batches", done, num_batches)

    if not all_logits:
        empty = torch.zeros(0)
        return empty, empty, empty
    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_weights)
