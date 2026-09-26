"""Raw span-type score extraction for conformal calibration.

Intercepts GLiNER's forward pass immediately after ``run_batch()``, before
sigmoid/threshold/decode, giving the full dense ``(B, L, K, C)`` candidate
span-score tensor. Reuses ``GLiNER.prepare_base_input`` /
``collate_batch`` / ``run_batch`` directly -- no custom tokenization or collation
logic, no core model changes.

Scope: span-mode uni-/bi-encoder models only (``UniEncoderSpanGLiNER``,
``BiEncoderSpanGLiNER`` -- the default ``span_mode="markerV0"`` architecture).
Token-mode, decoder, and relex variants apply ``threshold`` *inside* their
forward pass to prune candidate spans before returning scores
(``gliner/modeling/base.py``: ``get_span_representations`` ->
``extract_spans_from_tokens``, and the relex adjacency-selection paths), so
``run_batch()``'s output is not the full candidate universe for those
architectures -- calibrating against it would silently understate true
coverage. Verified by reading every ``forward()`` in ``gliner/modeling/base.py``:
the two span-mode classes never reference ``threshold``, so it is decode-only
for them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Sequence
from dataclasses import dataclass

import torch

_SPAN_MODE_CLASS_NAMES = {"UniEncoderSpanGLiNER", "BiEncoderSpanGLiNER"}


def _assert_span_mode_supported(model: Any) -> None:
    cls_name = type(model).__name__
    if cls_name not in _SPAN_MODE_CLASS_NAMES:
        raise NotImplementedError(
            f"ConformalGLiNER v1 only supports span-mode models "
            f"({sorted(_SPAN_MODE_CLASS_NAMES)}), got {cls_name!r}. Token-mode, "
            "decoder, and relex variants apply `threshold` inside their forward "
            "pass to prune candidate spans before returning scores, so "
            "run_batch()'s output is not the full candidate universe for those "
            "architectures. See docs/conformal.md for details."
        )


@dataclass
class RawScoreBatch:
    """Raw per-(span,type) scores for one collated batch, pre-sigmoid/threshold/decode.

    Attributes:
        logits: ``(B, L, K, C)`` raw span-mode scores, pre-sigmoid.
        id_to_classes: per-item ``{1-indexed class id: type string}`` maps
            (0 is reserved/unused, matching ``gliner/decoding/decoder.py``'s convention).
        tokens: per-item word-token lists, aligned with the ``(start, end)``
            word indices in each example's gold ``ner`` triples.
    """

    logits: torch.Tensor
    id_to_classes: List[Dict[int, str]]
    tokens: List[List[str]]


def extract_raw_scores(model: Any, examples: Sequence[Dict[str, Any]], labels: Sequence[str]) -> RawScoreBatch:
    """Run one forward pass and return dense pre-sigmoid span-type scores.

    Args:
        model: A span-mode ``GLiNER`` instance.
        examples: Pre-tokenized examples, ``{"tokenized_text": List[str], "ner": ...}``
            (the ``"ner"`` field is ignored here; use :func:`align_gold_scores` to pull
            out gold-span scores). Passing already-tokenized words (rather than raw
            text through ``model.prepare_batch``) is deliberate: it guarantees the
            word indices in ``examples[i]["ner"]`` line up exactly with the model's
            own span indexing, with no re-tokenization drift.
        labels: The fixed target label set to score every example against.

    Returns:
        RawScoreBatch with the dense score tensor and per-item bookkeeping.
    """
    _assert_span_mode_supported(model)
    if not examples:
        raise ValueError("No examples to score.")

    all_tokens = [ex["tokenized_text"] for ex in examples]
    input_x = model.prepare_base_input(all_tokens)
    batch = model.collate_batch(input_x, list(labels))
    model_output = model.run_batch(batch, threshold=0.0, move_to_device=True)

    logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)

    id_to_classes = batch["id_to_classes"]
    if not isinstance(id_to_classes, list):
        id_to_classes = [id_to_classes] * logits.shape[0]

    return RawScoreBatch(logits=logits, id_to_classes=id_to_classes, tokens=batch["tokens"])


def align_gold_scores(
    raw: RawScoreBatch,
    examples: Sequence[Dict[str, Any]],
) -> Tuple[List[float], List[str], List[int]]:
    """Pull nonconformity scores ``1 - sigmoid(logit)`` for every gold ``(span, type)`` pair.

    Args:
        raw: Output of :func:`extract_raw_scores` for the same ``examples``.
        examples: Same list passed to :func:`extract_raw_scores` (must match order/length).

    Returns:
        Tuple of parallel lists ``(scores, types, example_idx)``: nonconformity score,
        gold entity type, and the index into ``examples`` it came from. A gold span
        wider than ``max_width`` (not representable in the candidate universe at all --
        GLiNER structurally cannot ever predict it) gets score ``float("inf")`` --
        guaranteed non-conforming, guaranteed "missed" under
        risk-control, exactly the correct behavior for an unrepresentable entity, not
        a special case to filter out.
    """
    if len(examples) != len(raw.id_to_classes):
        raise ValueError(f"examples/raw batch size mismatch: {len(examples)} vs {len(raw.id_to_classes)}")

    probs = torch.sigmoid(raw.logits)
    _, L, K, C = probs.shape

    scores: List[float] = []
    types: List[str] = []
    example_idx: List[int] = []

    for i, ex in enumerate(examples):
        class_to_id = {v: k for k, v in raw.id_to_classes[i].items()}
        for start, end, etype in ex.get("ner", []):
            width_offset = end - start
            if etype not in class_to_id:
                continue  # type not in this batch's label set -- not calibratable from this call
            col = class_to_id[etype] - 1  # id_to_classes is 1-indexed (0 reserved)
            if not (0 <= start < L) or not (0 <= width_offset < K) or not (0 <= col < C):
                score = float("inf")
            else:
                score = 1.0 - probs[i, start, width_offset, col].item()
            scores.append(score)
            types.append(etype)
            example_idx.append(i)

    return scores, types, example_idx
