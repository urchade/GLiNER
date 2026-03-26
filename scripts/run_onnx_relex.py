"""
Low-level ONNX inference for UniEncoderSpanRelexGLiNER.

Dependencies:
    pip install onnxruntime transformers numpy

No GLiNER library required — only onnxruntime + transformers tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match values used during training / export)
# ──────────────────────────────────────────────────────────────────────────────
MAX_SPAN_WIDTH   = 12   # maximum entity span length in words
MAX_SEQ_LEN      = 512  # maximum tokenizer sequence length
ENT_TOKEN        = "<<ENT>>"
SEP_TOKEN        = "<<SEP>>"
REL_TOKEN        = "<<REL>>"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Entity:
    start: int          # character offset (inclusive)
    end: int            # character offset (exclusive)
    text: str
    label: str
    score: float


@dataclass
class Relation:
    head: Entity
    tail: Entity
    relation: str
    score: float


@dataclass
class InferenceResult:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Session loader
# ──────────────────────────────────────────────────────────────────────────────
def load_session(
    onnx_path: str | Path,
    use_gpu: bool = False,
) -> ort.InferenceSession:
    """Load an ONNX Runtime session.

    Args:
        onnx_path: Path to the .onnx model file.
        use_gpu:   Use CUDAExecutionProvider when True.

    Returns:
        Configured InferenceSession.
    """
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    print(f"[ONNX] Loaded model from {onnx_path}")
    print(f"[ONNX] Active providers: {session.get_providers()}")
    return session


def load_tokenizer(
    model_name_or_path: str | Path,
    cache_dir: str | None = None,
) -> AutoTokenizer:
    """Load tokenizer and register the three GLiNER special tokens.

    The tokens must match exactly what was used during training so that
    class_token_index and rel_token_index align with the exported model.

    Args:
        model_name_or_path: HuggingFace model id or local directory.
        cache_dir:           Optional cache directory.

    Returns:
        Tokenizer with special tokens added.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    new_tokens = [ENT_TOKEN, SEP_TOKEN, REL_TOKEN]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"[Tokenizer] Added {num_added} special tokens. Vocab size: {len(tokenizer)}")
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────
def _split_words(text: str) -> tuple[list[str], list[int]]:
    """Whitespace-split text; return tokens and their character start offsets.

    Args:
        text: Raw input string.

    Returns:
        (words, word_starts) where word_starts[i] is the char index of words[i].
    """
    words, starts = [], []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        starts.append(m.start())
    return words, starts


def _build_label_prefix(
    entity_types: list[str],
    relation_types: list[str],
    tokenizer: AutoTokenizer,
) -> tuple[list[int], int, int]:
    """Tokenize the label prefix that is prepended to every input.

    Format:
        <<ENT>> ent_type_1 <<ENT>> ent_type_2 ... <<SEP>>
        <<REL>> rel_type_1 <<REL>> rel_type_2 ... <<SEP>>

    Args:
        entity_types:   Ordered list of entity class names.
        relation_types: Ordered list of relation class names.
        tokenizer:      Tokenizer with special tokens registered.

    Returns:
        (prefix_ids, class_token_index, rel_token_index)
        class_token_index — vocab index of <<ENT>>
        rel_token_index   — vocab index of <<REL>>
    """
    ent_id = tokenizer.convert_tokens_to_ids(ENT_TOKEN)
    sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    rel_id = tokenizer.convert_tokens_to_ids(REL_TOKEN)

    prefix_ids: list[int] = []

    # Entity type tokens
    for etype in entity_types:
        prefix_ids.append(ent_id)
        prefix_ids.extend(tokenizer.encode(etype, add_special_tokens=False))
    prefix_ids.append(sep_id)

    # Relation type tokens (may be empty)
    for rtype in relation_types:
        prefix_ids.append(rel_id)
        prefix_ids.extend(tokenizer.encode(rtype, add_special_tokens=False))
    if relation_types:
        prefix_ids.append(sep_id)

    return prefix_ids, ent_id, rel_id


def _tokenize_single(
    text: str,
    entity_types: list[str],
    relation_types: list[str],
    tokenizer: AutoTokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Build all token-level arrays for one text.

    Returns a dict with keys:
        input_ids     : list[int]           — full token id sequence
        attention_mask: list[int]
        words_mask    : list[int]           — word index per token (0 = non-word)
        text_length   : int                 — number of words in text
        word_starts   : list[int]           — char offsets of each word
        words         : list[str]
        prefix_len    : int                 — number of prefix tokens (before [CLS] of text)
    """
    words, word_starts = _split_words(text)
    prefix_ids, _, _ = _build_label_prefix(entity_types, relation_types, tokenizer)

    # [CLS] + prefix + text words + [SEP]
    cls_id = tokenizer.cls_token_id
    sep_id_model = tokenizer.sep_token_id

    all_ids    : list[int] = [cls_id] + prefix_ids
    words_mask : list[int] = [0] * len(all_ids)   # 0 = not a word token

    word_subtoken_starts: list[int] = []  # first subtoken position for each word

    for word_idx, word in enumerate(words):
        sub_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(all_ids) + len(sub_ids) + 1 > max_seq_len:
            # Truncate: stop adding words
            break
        word_subtoken_starts.append(len(all_ids))
        for i, sid in enumerate(sub_ids):
            all_ids.append(sid)
            # Only the FIRST subtoken of a word gets a non-zero word index
            words_mask.append(word_idx + 1 if i == 0 else 0)

    all_ids.append(sep_id_model)
    words_mask.append(0)

    num_words = len(word_subtoken_starts)

    return {
        "input_ids":      all_ids,
        "attention_mask": [1] * len(all_ids),
        "words_mask":     words_mask,
        "text_length":    num_words,
        "word_starts":    word_starts[:num_words],
        "words":          words[:num_words],
        "prefix_len":     1 + len(prefix_ids),  # CLS + prefix tokens
    }


def _build_spans(
    num_words: int, max_span_width: int = MAX_SPAN_WIDTH
) -> tuple[np.ndarray, np.ndarray]:
    """Build a DENSE (num_words × max_span_width) span grid.

    The model's span_rep_layer reshapes its input as:
        (batch, num_words * max_span_width, hidden)
        → (batch, num_words, max_span_width, hidden)

    so span_idx MUST have exactly num_words * max_span_width rows — even for
    spans that extend past the end of the text (those are masked out).

    The triangular enumeration (only valid spans) produces 90 rows for
    13 words / width-12, while the model expects 13×12 = 156 rows,
    causing the "cannot reshape {2,90,768} → {2,13,12,768}" crash.

    Args:
        num_words:      Number of words in the text.
        max_span_width: Maximum span width in words (end is inclusive).

    Returns:
        span_idx  — shape (num_words * max_span_width, 2), dtype int64
                    each row is (start_word, end_word) — end may exceed text
        span_mask — shape (num_words * max_span_width,), dtype bool
                    True only when end_word < num_words
    """
    spans, mask = [], []
    for start in range(num_words):
        for width in range(max_span_width):   # width 0 = single-word span
            end = start + width
            valid = end < num_words
            # ScatterND validates ALL indices regardless of mask, so clamp
            # out-of-bounds ends to the last valid word index.
            spans.append((start, min(end, num_words - 1)))
            mask.append(valid)
    return np.array(spans, dtype=np.int64), np.array(mask, dtype=bool)


def _pad_batch(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Collate a list of single-sample dicts into padded numpy arrays.

    Args:
        samples: Output of _tokenize_single for each text in the batch.

    Returns:
        Dict of numpy arrays ready to feed into the ONNX session.
    """
    B = len(samples)

    max_seq  = max(len(s["input_ids"]) for s in samples)
    # Dense grid: num_words * max_span_width rows per sample.
    max_span = max(_build_spans(s["text_length"])[0].shape[0] for s in samples)

    input_ids      = np.zeros((B, max_seq),       dtype=np.int64)
    attention_mask = np.zeros((B, max_seq),       dtype=np.int64)
    words_mask     = np.zeros((B, max_seq),       dtype=np.int64)
    text_lengths   = np.zeros((B, 1),             dtype=np.int64)
    span_idx       = np.zeros((B, max_span, 2),   dtype=np.int64)
    span_mask      = np.zeros((B, max_span),      dtype=bool)

    for i, s in enumerate(samples):
        seq_len = len(s["input_ids"])
        input_ids[i, :seq_len]      = s["input_ids"]
        attention_mask[i, :seq_len] = s["attention_mask"]
        words_mask[i, :seq_len]     = s["words_mask"]
        text_lengths[i, 0]          = s["text_length"]

        spans, s_mask = _build_spans(s["text_length"])
        n = spans.shape[0]
        span_idx[i, :n, :]  = spans
        span_mask[i, :n]    = s_mask

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "words_mask":     words_mask,
        "text_lengths":   text_lengths,
        "span_idx":       span_idx,
        "span_mask":      span_mask,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _decode_entities(
    logits: np.ndarray,
    num_words: int,
    max_span_width: int,
    id_to_entity: dict[int, str],
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
) -> list[tuple[int, int, str, float]]:
    """Decode entity spans from model logits.

    logits shape : (num_words, max_span_width, num_ent_classes)
        The model computes scores = einsum("BLKD,BCD->BLKC", span_rep, prompts)
        so axis 0 = word start, axis 1 = span width (0 = single word), axis 2 = class.

    Returns list of (word_start, word_end, label, score).
    """
    probs = _sigmoid(logits)  # (num_words, max_span_width, num_ent_classes)

    candidates: list[tuple[int, int, str, float]] = []

    for start in range(min(num_words, probs.shape[0])):
        for width in range(min(max_span_width, probs.shape[1])):
            end = start + width  # inclusive end word index
            if end >= num_words:
                break  # wider spans for this start are all invalid

            if multi_label:
                for cls_i, label in id_to_entity.items():
                    score = float(probs[start, width, cls_i])
                    if score >= threshold:
                        candidates.append((start, end, label, score))
            else:
                best_cls = int(probs[start, width].argmax())
                best_score = float(probs[start, width, best_cls])
                if best_score >= threshold and best_cls in id_to_entity:
                    candidates.append((start, end, id_to_entity[best_cls], best_score))

    if flat_ner:
        candidates = _greedy_flat_ner(candidates)

    return candidates


def _greedy_flat_ner(
    candidates: list[tuple[int, int, str, float]],
) -> list[tuple[int, int, str, float]]:
    """Greedy non-overlapping span selection (highest score first).

    Args:
        candidates: List of (ws, we, label, score).

    Returns:
        Non-overlapping subset sorted by start position.
    """
    candidates = sorted(candidates, key=lambda x: -x[3])
    occupied: set[int] = set()
    selected = []
    for ws, we, label, score in candidates:
        span_positions = set(range(ws, we + 1))
        if not span_positions & occupied:
            selected.append((ws, we, label, score))
            occupied |= span_positions
    return sorted(selected, key=lambda x: x[0])


def _reconstruct_model_entity_spans(
    logits: np.ndarray,
    span_mask_2d: np.ndarray,
    num_words: int,
    max_span_width: int,
    selection_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Reconstruct the entity span list that the model selected internally.

    The model's select_span_target_embedding selects spans where
    sigmoid(scores).max(class_axis) > threshold, then packs them to the front
    using argsort(mask, descending=True).  rel_idx values reference positions
    in this packed list.

    Returns list of (word_start, word_end) in the same order the model uses.
    """
    probs = _sigmoid(logits)  # (L, K, C)
    L = min(num_words, probs.shape[0])
    K = min(max_span_width, probs.shape[1])

    # Flatten to (L*K,) max-class probability, matching model's flat ordering
    max_probs = probs[:L, :K].max(axis=-1)  # (L, K)

    # Build flat list of (flat_idx, start, end) for selected spans
    selected: list[tuple[int, int]] = []
    for start in range(L):
        for width in range(K):
            end = start + width
            if end >= num_words:
                break
            flat_idx = start * max_span_width + width
            if flat_idx < span_mask_2d.shape[0] and span_mask_2d[flat_idx]:
                if max_probs[start, width] > selection_threshold:
                    selected.append((start, end))

    return selected


def _decode_relations(
    rel_idx: np.ndarray,
    rel_logits: np.ndarray,
    rel_mask: np.ndarray,
    id_to_relation: dict[int, str],
    relation_threshold: float,
) -> list[tuple[int, int, str, float]]:
    """Decode relation predictions.

    rel_idx    : (num_pairs, 2)  — entity indices into model's internal entity list
    rel_logits : (num_pairs, num_rel_classes)
    rel_mask   : (num_pairs,)

    Returns list of (head_entity_idx, tail_entity_idx, relation_label, score).
    """
    probs = _sigmoid(rel_logits)  # (num_pairs, num_rel_classes)
    results = []

    for p_idx in range(rel_idx.shape[0]):
        if not rel_mask[p_idx]:
            continue
        head_idx = int(rel_idx[p_idx, 0])
        tail_idx = int(rel_idx[p_idx, 1])

        # Check each relation class (model uses 0-indexed classes)
        for cls_i in range(probs.shape[1]):
            score = float(probs[p_idx, cls_i])
            if score >= relation_threshold and cls_i in id_to_relation:
                results.append((head_idx, tail_idx, id_to_relation[cls_i], score))

    return results


def _spans_to_chars(
    word_spans: list[tuple[int, int, str, float]],
    word_starts: list[int],
    words: list[str],
) -> list[Entity]:
    """Map word-index spans back to character offsets.

    Args:
        word_spans:  List of (word_start, word_end, label, score).
        word_starts: Character offset of each word.
        words:       Word strings.

    Returns:
        List of Entity objects with character-level start/end.
    """
    entities = []
    for ws, we, label, score in word_spans:
        char_start = word_starts[ws]
        char_end   = word_starts[we] + len(words[we])
        entities.append(Entity(
            start=char_start,
            end=char_end,
            text="".join(w + " " for w in words[ws:we + 1]).rstrip(),
            label=label,
            score=score,
        ))
    return entities


# ──────────────────────────────────────────────────────────────────────────────
# Main inference function
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(
    texts: list[str],
    entity_types: list[str],
    relation_types: list[str],
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    threshold: float = 0.5,
    relation_threshold: float | None = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    batch_size: int = 8,
    max_span_width: int = MAX_SPAN_WIDTH,
    max_seq_len: int = MAX_SEQ_LEN,
) -> list[InferenceResult]:
    """Run entity + relation inference on a list of texts.

    Args:
        texts:              Input strings.
        entity_types:       Ordered entity class names (same order as during training).
        relation_types:     Ordered relation class names.
        session:            ONNX Runtime InferenceSession.
        tokenizer:          HuggingFace tokenizer with GLiNER special tokens.
        threshold:          Entity confidence threshold.
        relation_threshold: Relation confidence threshold (defaults to threshold).
        flat_ner:           Enforce non-overlapping entities.
        multi_label:        Allow multiple entity labels per span.
        batch_size:         Texts per ONNX call.
        max_span_width:     Max entity length in words.
        max_seq_len:        Max tokenizer sequence length.

    Returns:
        List of InferenceResult (one per input text).
    """
    if relation_threshold is None:
        relation_threshold = threshold

    # Build class index maps (0-based, matching label order)
    id_to_entity   = {i: t for i, t in enumerate(entity_types)}
    id_to_relation = {i: t for i, t in enumerate(relation_types)}

    results: list[InferenceResult] = [InferenceResult() for _ in texts]

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        # ── Tokenise ────────────────────────────────────────────────────────
        samples = [
            _tokenize_single(t, entity_types, relation_types, tokenizer, max_seq_len)
            for t in batch_texts
        ]
        feed = _pad_batch(samples)

        # ── ONNX forward ────────────────────────────────────────────────────
        # Try to get entity_spans if the model exports it (re-exported models);
        # fall back to 4 outputs for older exports.
        available_outputs = {o.name for o in session.get_outputs()}
        if "entity_spans" in available_outputs:
            output_names = ["logits", "rel_idx", "rel_logits", "rel_mask", "entity_spans"]
        else:
            output_names = ["logits", "rel_idx", "rel_logits", "rel_mask"]
        ort_outputs = session.run(
            output_names,
            {
                "input_ids":      feed["input_ids"],
                "attention_mask": feed["attention_mask"],
                "words_mask":     feed["words_mask"],
                "text_lengths":   feed["text_lengths"],
                "span_idx":       feed["span_idx"],
                "span_mask":      feed["span_mask"],
            },
        )
        batch_logits     = ort_outputs[0]  # (B, num_words, max_width, num_ent_cls)
        batch_rel_idx    = ort_outputs[1]  # (B, num_pairs, 2)
        batch_rel_logits = ort_outputs[2]  # (B, num_pairs, num_rel_cls)
        batch_rel_mask   = ort_outputs[3]  # (B, num_pairs)
        batch_entity_spans = ort_outputs[4] if len(ort_outputs) > 4 else None

        # ── Shape debug ─────────────────────────────────────────────────────
        W = 65
        print(f"\n{'─' * W}")
        print(f"  BATCH  offset={batch_start}  size={len(batch_texts)}")
        print(f"{'─' * W}")
        print(f"  {'tensor':<22} {'shape':<28} dtype")
        print(f"{'─' * W}")
        for name, arr in feed.items():
            print(f"  [IN]  {name:<20} {str(arr.shape):<28} {arr.dtype}")
        print(f"{'─' * W}")
        for name, arr, note in [
            ("logits",     batch_logits,     "(B, num_words, max_width, num_ent_classes)"),
            ("rel_idx",    batch_rel_idx,    "(B, num_pairs, 2)"),
            ("rel_logits", batch_rel_logits, "(B, num_pairs, num_rel_classes)"),
            ("rel_mask",   batch_rel_mask,   "(B, num_pairs)"),
        ]:
            print(f"  [OUT] {name:<20} {str(arr.shape):<28} {arr.dtype}  # {note}")
        print(f"{'─' * W}\n")

        # ── Decode per sample ───────────────────────────────────────────────
        for local_i, sample in enumerate(samples):
            global_i = batch_start + local_i
            num_words = sample["text_length"]

            # Entity decoding
            # logits shape: (num_words_padded, max_width, num_ent_classes)
            word_entities = _decode_entities(
                logits         = batch_logits[local_i],
                num_words      = num_words,
                max_span_width = max_span_width,
                id_to_entity   = id_to_entity,
                threshold      = threshold,
                flat_ner       = flat_ner,
                multi_label    = multi_label,
            )

            char_entities = _spans_to_chars(
                word_entities,
                sample["word_starts"],
                sample["words"],
            )
            results[global_i].entities = char_entities

            # Relation decoding  (requires mapping model's internal entity
            # indices back to our decoded entities)
            if relation_types:
                rel_pairs = _decode_relations(
                    rel_idx           = batch_rel_idx[local_i],
                    rel_logits        = batch_rel_logits[local_i],
                    rel_mask          = batch_rel_mask[local_i],
                    id_to_relation    = id_to_relation,
                    relation_threshold= relation_threshold,
                )

                # Get the model's internal entity span list.
                # If entity_spans is available from ONNX output, use it
                # directly; otherwise reconstruct from logits.
                if batch_entity_spans is not None:
                    es = batch_entity_spans[local_i]  # (E, 2)
                    model_entity_spans = [
                        (int(es[e, 0]), int(es[e, 1]))
                        for e in range(es.shape[0])
                        if not (es[e, 0] == 0 and es[e, 1] == 0 and e > 0)
                    ]
                else:
                    model_entity_spans = _reconstruct_model_entity_spans(
                        logits             = batch_logits[local_i],
                        span_mask_2d       = feed["span_mask"][local_i],
                        num_words          = num_words,
                        max_span_width     = max_span_width,
                        selection_threshold= 0.5,  # model uses default 0.5
                    )

                # Build mapping: model entity index → decoded entity index
                # by matching (start, end) boundaries
                decoded_boundary_to_idx: dict[tuple[int, int], int] = {}
                for idx, (ws, we, _, _) in enumerate(word_entities):
                    key = (ws, we)
                    if key not in decoded_boundary_to_idx:
                        decoded_boundary_to_idx[key] = idx

                model_to_decoded: dict[int, int] = {}
                for model_idx, (ms, me) in enumerate(model_entity_spans):
                    decoded_idx = decoded_boundary_to_idx.get((ms, me))
                    if decoded_idx is not None:
                        model_to_decoded[model_idx] = decoded_idx

                relations: list[Relation] = []
                for model_head, model_tail, rel_label, score in rel_pairs:
                    head_idx = model_to_decoded.get(model_head)
                    tail_idx = model_to_decoded.get(model_tail)
                    if head_idx is not None and tail_idx is not None:
                        relations.append(Relation(
                            head     = char_entities[head_idx],
                            tail     = char_entities[tail_idx],
                            relation = rel_label,
                            score    = score,
                        ))
                results[global_i].relations = relations

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────
def print_results(texts: list[str], results: list[InferenceResult]) -> None:
    for text, res in zip(texts, results):
        print(f"\n{'═' * 60}")
        print(f"TEXT: {text}\n")

        print("── Entities ──")
        if res.entities:
            for e in res.entities:
                print(f"  [{e.label:20s}] '{e.text}'  "
                      f"chars({e.start}–{e.end})  score={e.score:.3f}")
        else:
            print("  (none)")

        print("\n── Relations ──")
        if res.relations:
            for r in res.relations:
                print(f"  '{r.head.text}' ({r.head.label})"
                      f"  ──[{r.relation}]──▶"
                      f"  '{r.tail.text}' ({r.tail.label})"
                      f"  score={r.score:.3f}")
        else:
            print("  (none)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    ONNX_PATH        = "models/model.onnx"              # path to exported ONNX file
    TOKENIZER_SOURCE = "knowledgator/gliner-relex-large-v1.0"  # base encoder used for training

    ENTITY_TYPES   = ["Person", "Organization", "Location", "Date", "other"]
    RELATION_TYPES = ["works_for", "located in", "technologies", "created in", "money raised", "founded"]
    TEXTS = [
        "Elon Musk founded SpaceX in Hawthorne, California in 2002.",
        "Longevity AI drug discovery company Insilico Medicine has completed a $35 million Series D2 round, bringing the total raised in its Series D financing to $95 million. The new round was led by Prosperity7 Ventures, the diversified growth fund of Aramco Ventures, a subsidiary of Aramco, the world's leading integrated energy and chemicals company."
    ]

    # ── Load ──────────────────────────────────────────────────────────────────
    session   = load_session(ONNX_PATH, use_gpu=False)
    tokenizer = load_tokenizer(TOKENIZER_SOURCE)

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_inference(
        texts            = TEXTS,
        entity_types     = ENTITY_TYPES,
        relation_types   = RELATION_TYPES,
        session          = session,
        tokenizer        = tokenizer,
        threshold        = 0.3,
        relation_threshold = 0.5,
        flat_ner         = True,
        batch_size       = 4,
    )

    print_results(TEXTS, results)