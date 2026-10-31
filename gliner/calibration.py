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
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------


def assign_label_sets(
    data: Sequence[Dict[str, Any]],
    mode: str = "example",
    all_labels: Sequence[str] | None = None,
    seed: int = 42,
    key: str = "ner",
) -> List[Dict[str, Any]]:
    """Give every example a fixed ``{key}_labels`` list so collection and evaluation see the same labels.

    ``mode='dataset'`` gives every example ``all_labels``. ``mode='example'``
    gives an example its gold types plus as many random other types from
    ``all_labels`` (at least one). Examples that already have ``{key}_labels``
    keep them.
    """
    labels_key = f"{key}_labels"
    pool = list(all_labels) if all_labels is not None else dataset_entity_types(data, key)
    rng = random.Random(seed)
    out = []
    for example in data:
        if example.get(labels_key):
            out.append(example)
            continue
        if mode == "dataset":
            labels = list(pool)
        elif mode == "example":
            gold = sorted({str(ent[-1]) for ent in example.get(key, [])})
            gold_set = set(gold)
            others = [label for label in pool if label not in gold_set]
            negatives = rng.sample(others, min(len(others), max(1, len(gold))))
            labels = gold + negatives
            rng.shuffle(labels)
        else:
            raise ValueError(f"mode must be 'dataset' or 'example', got {mode!r}")
        out.append({**example, labels_key: labels})
    return out


# ---------------------------------------------------------------------------
# Calibration head: cached frozen outputs, fitting and evaluation
# ---------------------------------------------------------------------------


@dataclass
class CachedOutput:
    """Frozen-model outputs of one example, trimmed to its words and labels."""

    words: torch.Tensor  # (W, Dw)
    prompts: torch.Tensor  # (C, Dl)
    logits: torch.Tensor  # span: (W, K, C); token: (W, C, 3)
    targets: torch.Tensor  # same shape as logits
    mode: str = "span"  # "span" or "token"


@torch.inference_mode()
def collect_head_examples(
    model: Any,
    data: Sequence[Dict[str, Any]],
    batch_size: int = 8,
    store_dtype: torch.dtype = torch.float16,
    show_progress: bool = True,
) -> List[CachedOutput]:
    """Run the frozen model over labelled data and cache what the calibration head needs.

    Examples should carry ``ner_labels`` (see :func:`assign_label_sets`).
    Raw logits are cached even if a head is already attached.
    """
    from .modeling.calibrator import calibrator_mode  # noqa: PLC0415

    if getattr(model, "is_runtime_model", False):
        raise ValueError("Calibration needs a torch-runtime model; load it without ONNX/OpenVINO.")
    if not getattr(model, "supports_calibrator", True):
        raise NotImplementedError(f"{type(model).__name__} does not support a calibration head.")
    if getattr(model.config, "span_mode", None) == "token_level" and getattr(model.config, "represent_spans", False):
        raise NotImplementedError("Fitting a head for token models with represent_spans=True is not supported.")
    mode = calibrator_mode(model.config)

    collator = model._create_data_collator()
    device = model.device
    cached: List[CachedOutput] = []
    num_batches = math.ceil(len(data) / batch_size)
    for step, start in enumerate(range(0, len(data), batch_size), 1):
        examples = list(data[start : start + batch_size])
        batch = collator(examples)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        output = model(**batch)
        logits = getattr(output, "uncalibrated_logits", None)
        logits = output.logits if logits is None else logits
        labels = batch["labels"]
        num_words = output.mask.bool().sum(-1).tolist()
        num_classes = output.prompts_embedding_mask.bool().sum(-1).tolist()

        if mode == "span":
            B, L, K, _ = logits.shape
            labels = labels.view(B, L, K, -1)
        for i in range(logits.shape[0]):
            w, c = int(num_words[i]), int(num_classes[i])
            if w == 0 or c == 0:
                continue
            if mode == "span":
                lg, tg = logits[i, :w, :, :c], labels[i, :w, :, :c]
            else:
                lg, tg = logits[i, :w, :c], labels[i, :w, :c]
            cached.append(
                CachedOutput(
                    words=output.words_embedding[i, :w].to(store_dtype).cpu(),
                    prompts=output.prompts_embedding[i, :c].to(store_dtype).cpu(),
                    logits=lg.float().cpu(),
                    targets=tg.float().cpu(),
                    mode=mode,
                )
            )
        if show_progress and (step % 10 == 0 or step == num_batches):
            logger.info("cached %d/%d batches", step, num_batches)
    return cached


def _collate_cached(examples: Sequence[CachedOutput], device) -> Dict[str, torch.Tensor]:
    B = len(examples)
    W = max(ex.words.shape[0] for ex in examples)
    C = max(ex.prompts.shape[0] for ex in examples)
    first = examples[0]
    words = first.words.new_zeros(B, W, first.words.shape[1])
    prompts = first.prompts.new_zeros(B, C, first.prompts.shape[1])
    span_layout = first.mode == "span"
    if span_layout:  # (W, K, C)
        K = first.logits.shape[1]
        logits = torch.full((B, W, K, C), -1e4)
        targets = torch.zeros(B, W, K, C)
    else:  # (W, C, 3)
        logits = torch.full((B, W, C, 3), -1e4)
        targets = torch.zeros(B, W, C, 3)
    word_mask = torch.zeros(B, W, dtype=torch.bool)
    class_mask = torch.zeros(B, C, dtype=torch.bool)
    for i, ex in enumerate(examples):
        w, c = ex.words.shape[0], ex.prompts.shape[0]
        words[i, :w], prompts[i, :c] = ex.words, ex.prompts
        if span_layout:
            logits[i, :w, :, :c], targets[i, :w, :, :c] = ex.logits, ex.targets
        else:
            logits[i, :w, :c], targets[i, :w, :c] = ex.logits, ex.targets
        word_mask[i, :w], class_mask[i, :c] = True, True
    return {
        "logits": logits.to(device),
        "targets": targets.to(device),
        "words": words.to(device),
        "prompts": prompts.to(device),
        "word_mask": word_mask.to(device),
        "class_mask": class_mask.to(device),
    }


def _iterate_candidates(calibrator, examples, device, batch_size=16, order=None, predict=True):
    """Yield (calibrator output dict, candidate targets) per batch of cached examples."""
    order = list(range(len(examples))) if order is None else order
    for start in range(0, len(order), batch_size):
        batch = _collate_cached([examples[i] for i in order[start : start + batch_size]], device)
        args = (batch["logits"], batch["words"], batch["prompts"], batch["word_mask"], batch["class_mask"])
        out = calibrator(*args) if predict else calibrator.gather(*args)
        yield out, batch["targets"][out["index"]]


def _decision_channel(values: torch.Tensor) -> torch.Tensor:
    """The channel that decides a token-level pair (inside); identity for span pairs."""
    return values[:, 2] if values.dim() == 2 else values


def candidate_cells(z: torch.Tensor, y: torch.Tensor, num_bins: int) -> torch.Tensor:
    """(raw confidence bin over [0.5, 1]) * 2 + (raw decision at 0.5 is correct) per candidate."""
    p = torch.sigmoid(_decision_channel(z))
    t = _decision_channel(y)
    confidence = torch.maximum(p, 1 - p)
    bins = torch.clamp(((confidence - 0.5) * 2 * num_bins).long(), max=num_bins - 1)
    return bins * 2 + ((p >= 0.5) == (t >= 0.5)).long()


@torch.no_grad()
def candidate_entries(calibrator, examples, device, batch_size=16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw candidate logits and targets (token channels kept as columns), on CPU."""
    zs, ys = [], []
    for out, targets in _iterate_candidates(calibrator, examples, device, batch_size, predict=False):
        zs.append(out["z"].cpu())
        ys.append(targets.cpu())
    if not zs:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(zs), torch.cat(ys)


def balance_table(z: torch.Tensor, y: torch.Tensor, num_bins: int = 5, max_weight: float = 10.0) -> torch.Tensor:
    """Per-cell weights giving every non-empty (confidence bin, correct / wrong) cell equal loss mass.

    Weights are clipped to ``max_weight`` and renormalised so the total mass is unchanged.
    """
    cells = candidate_cells(z, y, num_bins)
    mass = torch.bincount(cells, minlength=2 * num_bins).float()
    target = mass.sum() / (mass > 0).sum().clamp(min=1)
    weights = torch.where(mass > 0, target / mass.clamp(min=1e-12), torch.zeros_like(mass)).clamp(max=max_weight)
    return weights * mass.sum() / (weights * mass).sum().clamp(min=1e-12)


def head_loss(out, targets, base_beta, base_bias=0.0, prior_weight=0.01, bias_prior_weight=0.01, weights=None):
    """Weighted mean over candidates of NLL(sigmoid(beta * z + b)) + priors. Returns (loss, nll)."""
    from .modeling.calibrator import GLiNERCalibrator  # noqa: PLC0415

    calibrated = GLiNERCalibrator.calibrated_candidate_logits(out)
    nll = F.binary_cross_entropy_with_logits(calibrated, targets.clamp(min=0), reduction="none")
    if nll.dim() == 2:
        nll = nll.mean(-1)
    beta, bias = out["beta"], out["bias"]
    prior = prior_weight * (beta.clamp_min(1e-6).log() - math.log(base_beta)) ** 2
    prior = prior + bias_prior_weight * (bias - base_bias) ** 2
    if weights is None:
        weights = torch.ones_like(nll)
    total = weights.sum().clamp(min=1e-12)
    return ((nll + prior) * weights).sum() / total, (nll * weights).sum() / total


def fit_calibrator(
    calibrator,
    train_examples: Sequence[CachedOutput],
    eval_examples: Sequence[CachedOutput],
    device,
    base_beta: float,
    base_bias: float = 0.0,
    prior_weight: float = 0.01,
    bias_prior_weight: float = 0.01,
    cell_weights: torch.Tensor | None = None,
    num_bins: int = 5,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 42,
    min_base_beta: float = 0.05,
    log_fn=print,
):
    """Train the head on cached frozen outputs, keeping the weights with the best held-out NLL.

    The head starts from global scaling (``base_beta``, ``base_bias``) and is
    pulled back towards it by the priors. ``cell_weights`` (from
    :func:`balance_table`) reweights training candidates by raw confidence
    bin and correctness.
    """
    if base_beta < min_base_beta:
        log_fn(f"beta_0={base_beta:.3g} is below {min_base_beta}; using {min_base_beta}")
        base_beta = min_base_beta
    if not calibrator.use_bias:
        base_bias = 0.0
    rng = random.Random(seed)
    calibrator.to(device)
    calibrator.reset_to_global_scaling(base_beta, base_bias)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=weight_decay)
    table = None if cell_weights is None else cell_weights.to(device)

    def held_out_nll():
        calibrator.eval()
        total, count = 0.0, 0.0
        with torch.no_grad():
            for out, targets in _iterate_candidates(calibrator, eval_examples, device, 32):
                if out["z"].shape[0] == 0:
                    continue
                _, nll = head_loss(out, targets, base_beta, base_bias)
                total += nll.item() * out["z"].shape[0]
                count += out["z"].shape[0]
        return total / max(count, 1)

    best = held_out_nll()
    best_state = {k: v.detach().clone() for k, v in calibrator.state_dict().items()}
    log_fn(f"epoch 0: held-out NLL {best:.5f} (global scaling, beta_0={base_beta:.4f}, b_0={base_bias:.4f})")
    bad_epochs = 0
    for epoch in range(1, num_epochs + 1):
        calibrator.train()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        running, seen = 0.0, 0
        for out, targets in _iterate_candidates(calibrator, train_examples, device, batch_size, order):
            if out["z"].shape[0] == 0:
                continue
            weights = None if table is None else table[candidate_cells(out["z"], targets, num_bins)]
            loss, _ = head_loss(out, targets, base_beta, base_bias, prior_weight, bias_prior_weight, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * out["z"].shape[0]
            seen += out["z"].shape[0]
        nll = held_out_nll()
        log_fn(f"epoch {epoch}: train loss {running / max(seen, 1):.5f}, held-out NLL {nll:.5f}")
        if nll < best - 1e-5:
            best, bad_epochs = nll, 0
            best_state = {k: v.detach().clone() for k, v in calibrator.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    calibrator.load_state_dict(best_state)
    calibrator.eval()
    return calibrator


@torch.no_grad()
def evaluate_head_candidates(calibrator, examples, device, base_beta=None, base_bias=0.0, n_bins=15):
    """Entry-level metrics on candidates for the raw model, global scaling and the head."""
    calibrator.eval()
    raw, head, ys = [], [], []
    for out, targets in _iterate_candidates(calibrator, examples, device, 32):
        if out["z"].shape[0] == 0:
            continue
        raw.append(out["z"].reshape(-1).cpu())
        head.append(calibrator.calibrated_candidate_logits(out).reshape(-1).cpu())
        ys.append(targets.reshape(-1).clamp(min=0).cpu())
    if not raw:
        return {}
    raw, head, ys = torch.cat(raw), torch.cat(head), torch.cat(ys)
    results = {
        "num_candidate_entries": int(ys.numel()),
        "uncalibrated": calibration_metrics(raw, ys, n_bins=n_bins),
        "calibrated": calibration_metrics(head, ys, n_bins=n_bins),
    }
    if base_beta is not None:
        results["global_scaling"] = calibration_metrics(
            raw, ys, temperature=1 / base_beta, bias=base_bias, n_bins=n_bins
        )
    return results


# ---------------------------------------------------------------------------
# Decoded-span reliability
# ---------------------------------------------------------------------------


def _gold_char_spans(tokens: Sequence[str], ner) -> set:
    starts, pos = [], 0
    for token in tokens:
        starts.append(pos)
        pos += len(token) + 1
    return {(starts[s], starts[e] + len(tokens[e]), str(label)) for s, e, label in ner}


def span_reliability(
    model: Any,
    data: Sequence[Dict[str, Any]],
    min_score: float = 0.05,
    n_bins: int = 10,
    threshold: float = 0.5,
    batch_size: int = 8,
    key: str = "ner",
) -> Dict[str, Any]:
    """Reliability of decoded spans: of spans with score in a bin, how many are correct.

    Runs normal inference (flat NER) on ``" ".join(tokenized_text)`` with each
    example's ``{key}_labels`` and a low threshold ``min_score``. A span is correct
    when its character offsets and label exactly match a gold entity. ECE is
    ``sum_b n_b / N * |mean score_b - accuracy_b|`` over spans scoring ``>= min_score``.
    """
    texts = [" ".join(ex["tokenized_text"]) for ex in data]
    labels = [list(ex[f"{key}_labels"]) for ex in data]
    preds = model.inference(texts, labels, threshold=min_score, flat_ner=True, batch_size=batch_size)

    scores, correct, num_gold = [], [], 0
    for example, entities in zip(data, preds, strict=True):
        gold = _gold_char_spans(example["tokenized_text"], example.get(key, []))
        num_gold += len(gold)
        for ent in entities:
            scores.append(ent["score"])
            correct.append((ent["start"], ent["end"], ent["label"]) in gold)

    s = torch.tensor(scores, dtype=torch.float64)
    c = torch.tensor(correct, dtype=torch.float64)
    bins_out, ece = [], 0.0
    if len(s):
        bins = torch.clamp((s * n_bins).long(), max=n_bins - 1)
        for b in range(n_bins):
            in_bin = bins == b
            n = int(in_bin.sum())
            if n == 0:
                bins_out.append({"bin": [b / n_bins, (b + 1) / n_bins], "count": 0})
                continue
            conf, acc = float(s[in_bin].mean()), float(c[in_bin].mean())
            ece += n / len(s) * abs(conf - acc)
            bins_out.append({"bin": [b / n_bins, (b + 1) / n_bins], "count": n, "mean_score": conf, "accuracy": acc})
    above = s >= threshold
    tp = float(c[above].sum()) if len(s) else 0.0
    precision = tp / max(int(above.sum()), 1)
    recall = tp / max(num_gold, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "ece": ece,
        "num_spans": len(s),
        "bins": bins_out,
        f"precision@{threshold}": precision,
        f"recall@{threshold}": recall,
        f"f1@{threshold}": f1,
    }
