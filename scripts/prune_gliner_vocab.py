"""
Vocabulary Pruning Engine for GLiNER multilingual models.

Reduces the word embedding matrix from a full multilingual vocabulary (~250k tokens)
to a target-language-specific subset. Achieves significant model size reduction and
CPU inference speedup with no F1 loss on the target language.

Usage:
    # From Wikipedia (requires: pip install datasets)
    python scripts/prune_gliner_vocab.py \
        --model_id knowledgator/gliner-bi-small-v1.0 \
        --dataset_for_vocab wikipedia \
        --output_dir results/pruned_en \
        --lang en

    # From a plain text file (one sentence per line)
    python scripts/prune_gliner_vocab.py \
        --model_id knowledgator/gliner-bi-small-v1.0 \
        --dataset_for_vocab /path/to/corpus.txt \
        --output_dir results/pruned_en \
        --top_k 30000
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner import GLiNER


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_corpus(source: str, lang: str, max_texts: int) -> list[str]:
    """Load a text corpus from Wikipedia or a plain .txt file."""
    path = Path(source)
    if path.exists():
        print(f"[corpus] Loading from file: {path}")
        texts = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return texts[:max_texts] if max_texts else texts

    if source.lower() == "wikipedia":
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Install 'datasets' to use Wikipedia: pip install datasets"
            ) from None
        print(f"[corpus] Streaming Wikipedia ({lang})…")
        ds = load_dataset(
            "wikipedia", f"20220301.{lang}",
            split="train", streaming=True, trust_remote_code=True,
        )
        texts: list[str] = []
        for sample in tqdm(ds, desc="Fetching articles", total=max_texts):
            # Take the first 2 000 chars of each article to avoid long-tail tokens
            texts.append(sample["text"][:2000])
            if len(texts) >= max_texts:
                break
        return texts

    raise ValueError(
        f"--dataset_for_vocab must be 'wikipedia' or a path to a .txt file. Got: {source!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token frequency analysis
# ─────────────────────────────────────────────────────────────────────────────

def _collect_active_ids(texts: list[str], tokenizer, top_k: Optional[int]) -> set[int]:
    """Tokenize corpus and return the set of active token IDs."""
    freq: Counter = Counter()
    for text in tqdm(texts, desc="Tokenising corpus"):
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=512
        )["input_ids"]
        freq.update(ids)

    if top_k is not None:
        active = {tid for tid, _ in freq.most_common(top_k)}
        print(f"[vocab]  Corpus active tokens (all): {len(freq):,} → top-{top_k}: {len(active):,}")
    else:
        active = set(freq.keys())
        print(f"[vocab]  Corpus active tokens: {len(active):,}")
    return active


# ─────────────────────────────────────────────────────────────────────────────
# Keep-set construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_keep_ids(tokenizer, active_ids: set[int]) -> list[int]:
    """
    Return a sorted list of token IDs to keep.

    Always includes:
      - Every token observed in the target-language corpus
      - All standard HuggingFace special tokens (PAD, UNK, CLS, SEP, MASK…)
      - All byte-fallback tokens (required for non-ASCII character coverage)
      - All explicitly added tokens (GLiNER's [FLERT], <<ENT>>, <<SEP>>, <<REL>>…)
    """
    keep: set[int] = set(active_ids)

    # 1. HuggingFace special-token attributes
    for attr in (
        "pad_token_id", "unk_token_id", "cls_token_id", "sep_token_id",
        "mask_token_id", "bos_token_id", "eos_token_id",
    ):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            keep.add(tid)

    # 2. Byte-fallback tokens (look like <0xNN> in the vocab)
    for piece, tid in tokenizer.get_vocab().items():
        if piece.startswith("<0x") and piece.endswith(">"):
            keep.add(tid)

    # 3. Explicitly added tokens (GLiNER specials and anything else)
    if hasattr(tokenizer, "added_tokens_encoder"):
        keep.update(tokenizer.added_tokens_encoder.values())
    if hasattr(tokenizer, "added_tokens_decoder"):
        keep.update(tokenizer.added_tokens_decoder.keys())

    # Clamp to valid range
    V = len(tokenizer)
    keep = {tid for tid in keep if 0 <= tid < V}

    keep_ids = sorted(keep)
    print(
        f"[vocab]  Keep set: {len(keep_ids):,} / {V:,} tokens "
        f"({len(keep_ids) / V * 100:.1f}% retained)"
    )
    return keep_ids


# ─────────────────────────────────────────────────────────────────────────────
# Embedding surgery
# ─────────────────────────────────────────────────────────────────────────────

def _slice_word_embeddings(
    bert_model: nn.Module,
    keep_tensor: torch.Tensor,
    pad_new_id: int,
    label: str = "encoder",
) -> int:
    """
    Replace word_embeddings with a sliced version containing only kept rows.

    Invariant: E_new[new_id] == E_old[old_id]  for every (old_id → new_id) mapping.
    Returns the new vocab size K.
    """
    old_embed: nn.Embedding = bert_model.embeddings.word_embeddings
    E_old = old_embed.weight.data           # (V, d)
    E_new = E_old[keep_tensor]              # (K, d)

    K, d = E_new.shape
    new_embed = nn.Embedding(K, d, padding_idx=pad_new_id)
    new_embed.weight = nn.Parameter(E_new)
    new_embed.weight.requires_grad_(old_embed.weight.requires_grad)

    bert_model.embeddings.word_embeddings = new_embed
    bert_model.config.vocab_size = K

    V = E_old.shape[0]
    print(f"[embed]  {label}: {V:,} → {K:,} tokens  ({(1 - K/V)*100:.1f}% reduction)")
    return K


# ─────────────────────────────────────────────────────────────────────────────
# Fast-tokenizer JSON surgery
# ─────────────────────────────────────────────────────────────────────────────

def _prune_tokenizer_json(
    tok_json_path: Path,
    keep_ids: list[int],
    old_to_new: dict[int, int],
) -> None:
    """
    Rebuild tokenizer.json with only the kept vocabulary entries.

    For Unigram (SentencePiece fast) tokenizers:
      - tok["model"]["vocab"] is a list where list-index == token ID
      - We keep only the base-vocab rows at positions in keep_ids
      - GLiNER added tokens are handled via the top-level added_tokens list

    The serialised file overwrites the original in-place (call after save_pretrained).
    """
    raw = tok_json_path.read_text(encoding="utf-8")
    tok_data: dict = json.loads(raw)

    model_sec = tok_data.get("model", {})
    model_type = model_sec.get("type", "")
    if model_type != "Unigram":
        print(
            f"[warn]  tokenizer.json model type is {model_type!r}, not 'Unigram'. "
            "Vocabulary surgery may be incomplete."
        )

    # ── Slice the base vocabulary list ────────────────────────────────────────
    old_vocab: list = model_sec.get("vocab", [])
    base_len = len(old_vocab)   # number of base vocab entries (excluding added tokens)

    # Only select IDs that exist in old_vocab (added tokens are handled separately)
    base_keep = [i for i in keep_ids if i < base_len]
    model_sec["vocab"] = [old_vocab[i] for i in base_keep]

    # Remap unk_id within the new base vocab
    old_unk = model_sec.get("unk_id", 0)
    model_sec["unk_id"] = old_to_new.get(old_unk, 0)
    tok_data["model"] = model_sec

    # ── Remap added_tokens explicit IDs ───────────────────────────────────────
    for entry in tok_data.get("added_tokens", []):
        old_id = entry.get("id")
        if old_id is not None and old_id in old_to_new:
            entry["id"] = old_to_new[old_id]

    # ── Remap post_processor special-token ID maps (CLS/SEP templates) ────────
    post_proc = tok_data.get("post_processor") or {}
    for tok_info in post_proc.get("special_tokens", {}).values():
        if not isinstance(tok_info, dict):
            continue
        old_id = tok_info.get("id")
        if old_id is not None and old_id in old_to_new:
            tok_info["id"] = old_to_new[old_id]
        tok_info["ids"] = [old_to_new.get(i, i) for i in tok_info.get("ids", [])]

    tok_json_path.write_text(
        json.dumps(tok_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[tok]    Wrote pruned tokenizer.json  ({len(model_sec['vocab']):,} base entries)")


def _prune_added_tokens_json(added_json_path: Path, old_to_new: dict[int, int]) -> None:
    """Remap token IDs in added_tokens.json (used by slow-tokenizer path)."""
    data: dict = json.loads(added_json_path.read_text(encoding="utf-8"))
    new_data = {
        tok_str: old_to_new[old_id]
        for tok_str, old_id in data.items()
        if old_id in old_to_new
    }
    added_json_path.write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pruning engine
# ─────────────────────────────────────────────────────────────────────────────

def prune_gliner_vocab(
    model_id: str,
    dataset_source: str,
    output_dir: str,
    top_k: Optional[int] = None,
    lang: str = "en",
    max_corpus_texts: int = 100_000,
) -> dict:
    """
    Prune a multilingual GLiNER model's vocabulary to a target-language subset.

    Args:
        model_id: HuggingFace model ID or local directory path.
        dataset_source: 'wikipedia' (requires datasets package) or path to a .txt file.
        output_dir: Directory to write the pruned model.
        top_k: Cap the corpus-active tokens to the top-K by frequency. None = keep all seen.
        lang: Wikipedia language code ('en', 'fr', 'de', …). Ignored for text files.
        max_corpus_texts: Maximum number of articles / lines to read from the corpus.

    Returns:
        dict with pruning statistics.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n[model]  Loading {model_id!r}…")
    gliner_model = GLiNER.from_pretrained(model_id)
    gliner_model.eval()

    tokenizer = gliner_model.data_processor.transformer_tokenizer
    V_orig = len(tokenizer)
    print(f"[model]  Original vocab size: {V_orig:,}")

    # ── 2. Tokenise corpus to find active tokens ───────────────────────────────
    texts = _load_corpus(dataset_source, lang, max_corpus_texts)
    print(f"[corpus] Using {len(texts):,} texts")
    active_ids = _collect_active_ids(texts, tokenizer, top_k)

    # ── 3. Build keep set & remapping ─────────────────────────────────────────
    keep_ids = _build_keep_ids(tokenizer, active_ids)
    K = len(keep_ids)
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep_ids)}
    keep_tensor = torch.tensor(keep_ids, dtype=torch.long)
    pad_new_id = old_to_new.get(tokenizer.pad_token_id or 0, 0)

    # ── 4. Slice text-encoder word embeddings ─────────────────────────────────
    text_bert = gliner_model.model.token_rep_layer.bert_layer.model
    _slice_word_embeddings(text_bert, keep_tensor, pad_new_id, label="text encoder")

    # ── 5. Slice labels-encoder word embeddings (BiEncoder only) ─────────────
    token_rep = gliner_model.model.token_rep_layer
    if hasattr(token_rep, "labels_encoder"):
        le_bert = token_rep.labels_encoder.model
        le_V = le_bert.config.vocab_size
        if le_V == V_orig:
            _slice_word_embeddings(le_bert, keep_tensor, pad_new_id, label="labels encoder")
        else:
            print(
                f"[warn]  Labels encoder vocab size ({le_V:,}) differs from text encoder "
                f"({V_orig:,}); skipping labels encoder pruning."
            )

    # ── 6. Update GLiNER config ───────────────────────────────────────────────
    gliner_model.config.vocab_size = K
    enc_cfg = getattr(gliner_model.config, "encoder_config", None)
    if enc_cfg is not None:
        enc_cfg.vocab_size = K
    le_cfg = getattr(gliner_model.config, "labels_encoder_config", None)
    if le_cfg is not None and getattr(le_cfg, "vocab_size", None) == V_orig:
        le_cfg.vocab_size = K

    old_cti = gliner_model.config.class_token_index
    if old_cti == -1:
        # BiEncoder models leave class_token_index=-1 (architecture doesn't use it)
        print(f"[config] class_token_index=-1 (BiEncoder sentinel, no remap needed)  "
              f"vocab_size: {V_orig:,} → {K:,}")
    elif old_cti not in old_to_new:
        raise RuntimeError(
            f"class_token_index={old_cti} (GLiNER's entity marker token) is missing from "
            "the keep set — this is a bug in keep-set construction."
        )
    else:
        gliner_model.config.class_token_index = old_to_new[old_cti]
        print(
            f"[config] class_token_index: {old_cti} → {old_to_new[old_cti]}"
            f"  (vocab_size: {V_orig:,} → {K:,})"
        )

    # ── 7. Save model (weights carry sliced embedding; config has new K) ───────
    print(f"\n[save]   Writing pruned model to {out_path}…")
    gliner_model.save_pretrained(out_path)

    # ── 8. Rebuild tokenizer.json (save_pretrained wrote the original) ─────────
    tok_json = out_path / "tokenizer.json"
    if not tok_json.exists():
        print(
            "[warn]  tokenizer.json not found after save_pretrained. "
            "Fast tokenizer will not be available for the pruned model."
        )
    else:
        _prune_tokenizer_json(tok_json, keep_ids, old_to_new)

    added_json = out_path / "added_tokens.json"
    if added_json.exists():
        _prune_added_tokens_json(added_json, old_to_new)
        print("[tok]    Updated added_tokens.json")

    # ── 9. Summary ────────────────────────────────────────────────────────────
    d = text_bert.config.hidden_size
    orig_params = V_orig * d
    new_params  = K * d
    reduction   = (1 - new_params / orig_params) * 100

    print(f"\n{'═'*55}")
    print(f"  Original vocab     : {V_orig:>10,}")
    print(f"  Pruned  vocab      : {K:>10,}  ({K/V_orig*100:.1f}% retained)")
    print(f"  Embedding reduction: {reduction:.1f}%")
    print(f"  Output             : {out_path}")
    print(f"{'═'*55}\n")

    return {
        "orig_vocab_size":     V_orig,
        "pruned_vocab_size":   K,
        "retention_pct":       round(K / V_orig * 100, 2),
        "embed_reduction_pct": round(reduction, 2),
        "output_dir":          str(out_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune a GLiNER multilingual model's vocabulary to a target-language subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id", required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset_for_vocab", required=True,
        help="'wikipedia' or path to a plain text file (one sentence per line)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save the pruned model",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Keep only the top-K most frequent corpus tokens (default: keep ALL seen tokens)",
    )
    parser.add_argument(
        "--lang", default="en",
        help="Wikipedia language code, e.g. en, fr, de (ignored for text files)",
    )
    parser.add_argument(
        "--max_corpus_texts", type=int, default=100_000,
        help="Maximum number of texts / lines to read from the corpus",
    )
    args = parser.parse_args()

    prune_gliner_vocab(
        model_id=args.model_id,
        dataset_source=args.dataset_for_vocab,
        output_dir=args.output_dir,
        top_k=args.top_k,
        lang=args.lang,
        max_corpus_texts=args.max_corpus_texts,
    )


if __name__ == "__main__":
    main()
