"""
Joint NER + Relation Extraction training script for GLiNER-Relex.

Trains a UniEncoderSpanRelexGLiNER model that performs both named entity
recognition and relation extraction in a single forward pass.

Training data format (JSON Lines or JSON list):
    {
      "tokenized_text": ["Apple", "was", "founded", "by", "Steve", "Jobs"],
      "ner": [
        [0, 0, "organization"],
        [4, 5, "person"]
      ],
      "relations": [
        [4, 5, "person", 0, 0, "organization", "founded_by"]
      ]
    }

Each relation tuple: [head_start, head_end, head_type, tail_start, tail_end, tail_type, relation_type]

Usage:
    python scripts/train_relex.py \
        --model_id microsoft/deberta-v3-small \
        --train_data data/conll04_train.json \
        --eval_data  data/conll04_dev.json \
        --output_dir results/relex_conll04 \
        --max_steps 5000 \
        --batch_size 8
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner import GLiNER
from gliner.config import UniEncoderSpanRelexConfig
from gliner.training import TrainingArguments


def _load_jsonl(path: str) -> list:
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a joint NER + Relation Extraction GLiNER model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", default="microsoft/deberta-v3-small",
                        help="HuggingFace encoder ID or local path")
    parser.add_argument("--train_data", required=True,
                        help="Path to training data (.json or .jsonl)")
    parser.add_argument("--eval_data", default=None,
                        help="Path to evaluation data (.json or .jsonl)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=384,
                        help="Maximum sequence length")
    parser.add_argument("--max_width", type=int, default=12,
                        help="Maximum entity span width (tokens)")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="GLiNER hidden projection size")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log metrics every N steps")
    # Hard negative sampling
    parser.add_argument("--hard_negative_ratio", type=float, default=0.0,
                        help="Fraction of hard (semantically similar) negatives. 0=random only.")
    # Contrastive loss
    parser.add_argument("--contrastive_coef", type=float, default=0.0,
                        help="Weight for auxiliary contrastive loss. 0=disabled.")
    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true",
                        help="Enable curriculum learning (easy examples first)")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[data]  Loading training data from {args.train_data}…")
    train_data = _load_jsonl(args.train_data)
    eval_data  = _load_jsonl(args.eval_data) if args.eval_data else train_data[:100]
    print(f"[data]  Train: {len(train_data):,} examples  |  Eval: {len(eval_data):,} examples")

    # ── Build config ──────────────────────────────────────────────────────
    config = UniEncoderSpanRelexConfig(
        model_name=args.model_id,
        max_len=args.max_len,
        max_width=args.max_width,
        hidden_size=args.hidden_size,
    )

    # ── Instantiate model ─────────────────────────────────────────────────
    from gliner.model import UniEncoderSpanRelexGLiNER  # noqa: PLC0415
    print(f"[model] Initialising UniEncoderSpanRelexGLiNER from {args.model_id}…")
    model = UniEncoderSpanRelexGLiNER.load_from_config(
        config,
        backbone_from_pretrained=True,
    )

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(out_path),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="none",
        use_cpu=True,
        hard_negative_ratio=args.hard_negative_ratio,
        contrastive_loss_coef=args.contrastive_coef,
        use_curriculum=args.use_curriculum,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("[train] Starting training…")
    model.train_model(
        train_data,
        eval_data,
        training_args=training_args,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    model.save_pretrained(str(out_path / "final"))
    print(f"[done]  Model saved to {out_path / 'final'}")

    # ── Quick inference demo ───────────────────────────────────────────────
    print("\n[demo]  Loading saved model for quick sanity check…")
    loaded = UniEncoderSpanRelexGLiNER.from_pretrained(str(out_path / "final"))
    loaded.eval()

    text   = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    labels = ["organization", "person", "location"]
    rels   = ["founded_by", "located_in"]

    entities, relations = loaded.predict_relations(text, labels, rels, threshold=0.3)
    print(f"  Entities:  {[(e['text'], e['label']) for e in entities]}")
    print(f"  Relations: {[(r['head']['text'], r['relation'], r['tail']['text']) for r in relations]}")


if __name__ == "__main__":
    main()
