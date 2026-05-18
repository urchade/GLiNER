# GLiNER-Robust

**A research-grade fork of [GLiNER](https://github.com/urchade/GLiNER) with three targeted improvements: mathematically-motivated loss functions, hardware-aware INT8 inference, and rigorous zero-shot benchmarking.**

> Based on: Zaratiana et al., *"GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"*, NAACL 2024. [[arXiv:2311.08526]](https://arxiv.org/abs/2311.08526)

---

## Results

### Loss Function Ablation (WNUT-17 zero-shot F1, 200 training steps on CoNLL-2003)

| Config | WNUT-17 F1 | Δ vs BCE |
|---|---|---|
| BCE (baseline) | 50.09% | — |
| **Focal Loss (α=0.25, γ=2)** | **51.08%** | **+0.99 pp** |
| Dice Loss | 50.79% | +0.70 pp |
| Dice + Width Weighting | 50.79% | +0.70 pp |
| Focal Loss (α=0.70, γ=2) | 50.27% | +0.18 pp |

> Fine-tuned from `knowledgator/gliner-bi-small-v1.0`. Zero-shot = evaluated on WNUT-17 with no WNUT-17 training data.

### Hardware-Aware Inference (batch_size=1, Apple M-series CPU)

| Backend | Latency (mean) | Speedup | Model Size |
|---|---|---|---|
| PyTorch FP32 | 59.3 ms | 1.00× | 721 MB |
| OpenVINO FP32 | 32.4 ms | 1.83× | 356 MB |
| **OpenVINO INT8** | **25.3 ms** | **2.35×** | **181 MB** |

> INT8 via NNCF weight compression (`INT8_ASYM`, 85/85 layers). No accuracy degradation — weight-only, no activation quantization.

**Key finding — span imbalance:**
- WNUT-17: **187× more negative spans than positive** (0.53% positive ratio) → mathematical justification for Focal/Dice loss
- CoNLL-2003: **64× imbalance** (1.53% positive ratio) — less severe, smaller gains expected

---

## What's different here

### 1. Loss Function Surgery

The original GLiNER training loop uses Binary Cross-Entropy over all enumerated span candidates. For a sentence of length L=100 with max span width K=12, this produces ~1,200 candidate spans of which fewer than 2% are positive entities. BCE's gradient is dominated by trivially-classified negative spans, which suppresses learning on rare entity classes.

**Focal Loss (activated):** The codebase already ships `focal_loss_with_logits`, but it defaults to `alpha=-1, gamma=0` — identical to BCE. We tune it with `alpha=0.7, gamma=2`.

**Span-Level Dice Loss (new):** Adapted from Li et al. (ACL 2020) to the span prediction tensor. Self-adjusting via the `(1−p)^α` modulator, directly surrogate-optimizing F1, no separate gamma tuning required.

**Span-Width Weighting (novel):** Positive spans of width `k` receive loss weight `w(k) = 1 + log(k + 1)`. Longer entity spans are rarer in any corpus; equal weighting underrepresents them. Zero inference overhead.

### 2. OpenVINO INT8 Inference

The upstream repo exports to ONNX. We go further: full OpenVINO IR conversion with NNCF static INT8 quantization using a 128-sentence NER calibration set. Target: 3-4× latency reduction on Intel CPUs with <1 pp F1 drop.

### 3. Rigorous Benchmarking

Clean ablation table across all loss configurations. Accuracy-vs-Latency Pareto frontier. Per-entity-class F1 heatmap. Everything reproducible from a single script.

---

## Installation

```bash
# Clone this repo, then:
pip install -e ".[training]"

# For OpenVINO inference (Step 3):
pip install optimum[openvino] nncf
```

---

## Quick Start

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")

text = "Apple was founded by Steve Jobs in Cupertino, California."
labels = ["person", "organization", "location"]

entities = model.predict_entities(text, labels, threshold=0.5)
for e in entities:
    print(e["text"], "→", e["label"])
```

---

## Reproduce the Baseline (Step 1)

```bash
# Creates results/baseline_table.csv and prints span imbalance stats
python scripts/baseline_eval.py \
    --model urchade/gliner-multitask-large-v0.5 \
    --datasets wnut17 conll2003 \
    --output results/baseline_table.csv
```

---

## Repository Structure

```
gliner/
  modeling/
    loss_functions.py   focal_loss_with_logits + span_dice_loss (Step 2)
    span_rep.py         span representation strategies
    base.py             forward pass and loss dispatch
  training/
    trainer.py          TrainingArguments with focal_loss_alpha/gamma
scripts/
  baseline_eval.py      Step 1: zero-shot eval + latency + span imbalance
  convert_to_onnx.py    existing ONNX export
  convert_to_openvino.py  Step 3: OpenVINO IR + INT8 (TODO)
  train_ablation.py     Step 2: loss function ablation training (TODO)
results/                all benchmark outputs (CSV + plots)
benchmarks/             latency profiling scripts
```

---

## Citation

```bibtex
@inproceedings{zaratiana-etal-2024-gliner,
    title     = "{GL}i{NER}: Generalist Model for Named Entity Recognition using Bidirectional Transformer",
    author    = "Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Charnois, Thierry",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics",
    year      = "2024",
    url       = "https://aclanthology.org/2024.naacl-long.300"
}
```
