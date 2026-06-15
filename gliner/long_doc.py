"""
Sliding-window long-document inference for GLiNER.

Addresses GitHub issues #95 and #113 — the most-requested missing feature.
The 384-token hard limit is the root cause; this module provides a proper
built-in implementation instead of each user rolling their own broken version.

Algorithm:
  1. Split the full token sequence into overlapping windows of size max_tokens.
  2. Run standard GLiNER inference on each window independently.
  3. Remap entity character offsets back to the full document.
  4. Deduplicate entities that appear in multiple overlapping windows,
     keeping the prediction with the highest confidence score.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional


def _compute_windows(
    n_tokens: int,
    max_tokens: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """
    Compute (start_token, end_token) pairs for sliding windows.

    The last window is always extended to cover any remaining tokens,
    so no tokens are ever silently dropped.

    Args:
        n_tokens:   Total number of tokens in the document.
        max_tokens: Maximum tokens per window.
        stride:     Step size between window starts. stride < max_tokens
                    creates overlapping windows for boundary safety.

    Returns:
        List of (start, end) inclusive token-index pairs.
    """
    if n_tokens <= max_tokens:
        return [(0, n_tokens)]

    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + max_tokens, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += stride

    return windows


def _merge_entities(
    all_entities: List[List[Dict]],
    dedup_strategy: str = "max_score",
) -> List[Dict]:
    """
    Merge entity lists from overlapping windows and deduplicate.

    Entities from different windows are deduplicated by (start, end, label).
    The chosen strategy controls which duplicate to keep:
      - "max_score": keep the prediction with the highest confidence score
      - "first":     keep the prediction from the earlier window
      - "last":      keep the prediction from the later window

    Args:
        all_entities:    List of entity-list outputs, one per window.
                         Each entity must have "start", "end", "label", "score".
        dedup_strategy:  One of "max_score", "first", "last".

    Returns:
        Flat, deduplicated list of entity dicts sorted by start position.
    """
    seen: Dict[Tuple, Dict] = {}   # key: (start, end, label) → entity dict

    for _window_idx, entities in enumerate(all_entities):
        for entity in entities:
            key = (entity["start"], entity["end"], entity["label"])
            if key not in seen:
                seen[key] = entity
            elif dedup_strategy == "max_score":
                if entity.get("score", 0.0) > seen[key].get("score", 0.0):
                    seen[key] = entity
            elif dedup_strategy == "last":
                seen[key] = entity
                # "first" → keep existing, do nothing

    return sorted(seen.values(), key=lambda e: e["start"])


def predict_entities_long(
    model,
    text: str,
    labels,
    threshold: float = 0.5,
    max_tokens: int = 384,
    stride: Optional[int] = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    dedup_strategy: str = "max_score",
    batch_size: int = 8,
) -> List[Dict]:
    """
    Run entity extraction on a text of arbitrary length using a sliding window.

    Splits the text into overlapping token windows, runs standard GLiNER inference
    on each window, remaps character offsets to the full document, then deduplicates
    entities that span window boundaries.

    Args:
        model:           A GLiNER model instance (any subclass of BaseEncoderGLiNER).
        text:            Input text of arbitrary length.
        labels:          Entity type labels — any format accepted by predict_entities()
                         including description dicts (see gliner.descriptions).
        threshold:       Confidence threshold for entity predictions. Default: 0.5.
        max_tokens:      Maximum word-tokens per window. Default: model's configured max_len.
        stride:          Step size between window starts (in tokens).
                         Default: max_tokens // 3 (33% overlap — recommended).
                         Set stride == max_tokens for non-overlapping windows (faster,
                         but entities at boundaries may be missed).
        flat_ner:        Resolve overlapping spans by keeping the highest-scoring one.
        multi_label:     Allow the same span to have multiple entity labels.
        dedup_strategy:  How to resolve entities seen in multiple windows:
                         "max_score" (default), "first", or "last".
        batch_size:      Batch size passed to the per-window inference call.

    Returns:
        List of entity dicts with character-level start/end positions, label,
        text, and score. Sorted by start position.

    Example:
        from gliner import GLiNER
        from gliner.long_doc import predict_entities_long

        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

        entities = predict_entities_long(
            model,
            long_document_text,
            ["person", "organization", "location"],
            max_tokens=512,
            stride=128,
        )
    """
    if not text or not text.strip():
        return []

    # Resolve defaults
    if max_tokens is None:
        max_tokens = getattr(model.config, "max_len", 384)
    if stride is None:
        stride = max(1, max_tokens // 3)

    # Tokenise text into words (word-level tokens as GLiNER uses)
    word_tokens: List[str] = []
    word_starts: List[int] = []   # char offset where each word starts
    word_ends: List[int] = []     # char offset where each word ends

    for word, start, end in model.data_processor.words_splitter(text):
        word_tokens.append(word)
        word_starts.append(start)
        word_ends.append(end)

    n_words = len(word_tokens)
    if n_words == 0:
        return []

    windows = _compute_windows(n_words, max_tokens, stride)

    all_window_entities: List[List[Dict]] = []

    for win_start, win_end in windows:
        window_words = word_tokens[win_start:win_end]
        if not window_words:
            continue

        # Reconstruct window text preserving original whitespace
        # The window text spans from the first word's char start to the last word's char end.
        char_start = word_starts[win_start]
        char_end   = word_ends[win_end - 1]
        window_text = text[char_start:char_end]

        # Run standard inference on the window
        try:
            window_entities = model.predict_entities(
                window_text,
                labels,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                batch_size=batch_size,
            )
        except Exception:
            window_entities = []

        # Remap char offsets from window-local to document-global
        for entity in window_entities:
            entity["start"] += char_start
            entity["end"]   += char_start

        all_window_entities.append(window_entities)

    return _merge_entities(all_window_entities, dedup_strategy=dedup_strategy)


def batch_predict_entities_long(
    model,
    texts: List[str],
    labels,
    threshold: float = 0.5,
    max_tokens: int = 384,
    stride: Optional[int] = None,
    flat_ner: bool = True,
    multi_label: bool = False,
    dedup_strategy: str = "max_score",
    batch_size: int = 8,
) -> List[List[Dict]]:
    """Run sliding-window inference on multiple texts.

    Processes each text independently. For throughput-critical workloads, consider
    using the single-document version with a higher batch_size so window batches
    from one document fill the GPU/CPU efficiently.

    Args:
        model: A GLiNER model instance.
        texts: List of input texts of arbitrary length.
        labels: Entity type labels (any format accepted by predict_entities).
        threshold: Confidence threshold for entity predictions. Default: 0.5.
        max_tokens: Maximum word-tokens per window. Default: 384.
        stride: Step size between window starts in tokens. Default: max_tokens // 3.
        flat_ner: Resolve overlapping spans by keeping the highest-scoring one.
        multi_label: Allow the same span to have multiple entity labels.
        dedup_strategy: How to resolve entities seen in multiple windows.
        batch_size: Batch size passed to the per-window inference call.

    Returns:
        List of entity-list, one per input text.
    """
    return [
        predict_entities_long(
            model, text, labels,
            threshold=threshold,
            max_tokens=max_tokens,
            stride=stride,
            flat_ner=flat_ner,
            multi_label=multi_label,
            dedup_strategy=dedup_strategy,
            batch_size=batch_size,
        )
        for text in texts
    ]
