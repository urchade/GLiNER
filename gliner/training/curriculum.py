"""
Curriculum Learning sampler for GLiNER NER training.

Progressive curricula — training on easy examples first, gradually introducing harder ones —
consistently improve final F1 across multiple 2024-2025 NER papers. This module provides:

  - SpanDifficultyScorer: scores each training example by entity type rarity,
    span length, span density, and label-set diversity.
  - CurriculumSampler: a PyTorch Sampler that starts with the easiest fraction of
    the dataset and linearly expands to the full dataset over a configurable number
    of epochs.

Usage:
    from gliner.training.curriculum import SpanDifficultyScorer, CurriculumSampler

    scorer = SpanDifficultyScorer()
    scorer.fit(train_dataset)

    sampler = CurriculumSampler(
        train_dataset,
        scorer,
        start_pct=0.3,         # begin with easiest 30%
        ramp_epochs=5,          # reach full dataset by epoch 5
    )

    # In your training loop, call before each epoch:
    sampler.set_epoch(epoch)

Or via TrainingArguments (integrates automatically when use_curriculum=True):
    args = TrainingArguments(
        use_curriculum=True,
        curriculum_start_pct=0.3,
        curriculum_ramp_epochs=5,
        ...
    )
"""

from __future__ import annotations

import random
from typing import Dict, List, Iterator, Optional
from collections import Counter

from torch.utils.data import Sampler

# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

class SpanDifficultyScorer:
    """
    Assigns a difficulty score in [0, 1] to each training example.

    Difficulty is a weighted combination of four signals:

    1. type_rarity  (default weight 0.40) — how rare the entity types are
       in the overall training set. Rare types → harder to learn.

    2. span_length  (default weight 0.20) — average width of entity spans,
       normalised by max_width. Longer spans → harder boundary decisions.

    3. span_density (default weight 0.20) — number of entity spans divided
       by the number of words in the example. Dense examples → more
       ambiguous, harder to predict.

    4. label_diversity (default weight 0.20) — number of distinct entity
       types in the example, normalised by the global max. Examples with
       many different types → harder because of larger label-set confusion.

    All four components are independently normalised to [0, 1] across the
    training set before weighting, so changing one weight doesn't distort
    the scale of the others.

    Args:
        type_rarity_weight:    Weight for entity type rarity signal.
        span_length_weight:    Weight for average span length signal.
        span_density_weight:   Weight for entity span density signal.
        label_diversity_weight: Weight for label-set diversity signal.
        max_width:             Maximum span width (for normalisation). Default 12.
    """

    def __init__(
        self,
        type_rarity_weight:     float = 0.40,
        span_length_weight:     float = 0.20,
        span_density_weight:    float = 0.20,
        label_diversity_weight: float = 0.20,
        max_width: int = 12,
    ) -> None:
        self.weights = {
            "type_rarity":     type_rarity_weight,
            "span_length":     span_length_weight,
            "span_density":    span_density_weight,
            "label_diversity": label_diversity_weight,
        }
        self.max_width = max_width
        self._scores: Optional[List[float]] = None
        self._sorted_indices: Optional[List[int]] = None

    def fit(self, dataset: List[Dict]) -> None:
        """
        Compute and cache difficulty scores for all examples.

        Args:
            dataset: List of training examples in GLiNER format.
                     Each item should have "tokenized_text" and "ner" keys.
        """
        # ── Pass 1: global type frequency count ──────────────────────────
        type_freq: Counter = Counter()
        for item in dataset:
            for ann in item.get("ner", []):
                type_freq[ann[-1]] += 1

        total_anns = sum(type_freq.values()) or 1

        def type_rarity(item: Dict) -> float:
            """Higher when item contains rare entity types."""
            anns = item.get("ner", [])
            if not anns:
                return 0.0
            # Rarity of each type = 1 - (freq / total_anns); mean over types in item
            return sum(1.0 - type_freq[a[-1]] / total_anns for a in anns) / len(anns)

        def avg_span_len(item: Dict) -> float:
            """Average entity span width, normalised by max_width."""
            anns = item.get("ner", [])
            if not anns:
                return 0.0
            widths = [(a[1] - a[0] + 1) for a in anns]
            return sum(widths) / (len(widths) * self.max_width)

        def span_density(item: Dict) -> float:
            """Number of entity spans / number of words."""
            n_words = len(item.get("tokenized_text", [])) or 1
            return len(item.get("ner", [])) / n_words

        def label_diversity(item: Dict) -> float:
            """Number of distinct entity types in this example."""
            return len({a[-1] for a in item.get("ner", [])})

        # ── Pass 2: compute raw component values ──────────────────────────
        raw: Dict[str, List[float]] = {k: [] for k in self.weights}
        for item in dataset:
            raw["type_rarity"].append(type_rarity(item))
            raw["span_length"].append(avg_span_len(item))
            raw["span_density"].append(span_density(item))
            raw["label_diversity"].append(label_diversity(item))

        # ── Pass 3: normalise each component to [0, 1] ───────────────────
        def _normalise(vals: List[float]) -> List[float]:
            lo, hi = min(vals), max(vals)
            if hi == lo:
                return [0.0] * len(vals)
            return [(v - lo) / (hi - lo) for v in vals]

        normed = {k: _normalise(raw[k]) for k in raw}

        # ── Pass 4: weighted sum ──────────────────────────────────────────
        n = len(dataset)
        self._scores = [
            sum(self.weights[k] * normed[k][i] for k in self.weights)
            for i in range(n)
        ]

        # Pre-sort indices from easiest (lowest score) to hardest
        self._sorted_indices = sorted(range(n), key=lambda i: self._scores[i])

    @property
    def scores(self) -> List[float]:
        if self._scores is None:
            raise RuntimeError("Call fit() before accessing scores.")
        return self._scores

    @property
    def sorted_indices(self) -> List[int]:
        if self._sorted_indices is None:
            raise RuntimeError("Call fit() before accessing sorted_indices.")
        return self._sorted_indices


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------

class CurriculumSampler(Sampler):
    """
    Progressive curriculum sampler.

    In epoch 0 (or before calling set_epoch), samples from only the easiest
    `start_pct` fraction of examples. By epoch `ramp_epochs`, the full
    dataset is accessible. After that, standard random sampling is used.

    Args:
        dataset:      The training dataset (list or indexable).
        scorer:       A fitted SpanDifficultyScorer.
        start_pct:    Fraction of easiest examples to start with. Default: 0.30.
        ramp_epochs:  Number of epochs to linearly ramp up to 100%. Default: 5.
        seed:         Random seed for reproducibility.

    Example:
        sampler = CurriculumSampler(dataset, scorer, start_pct=0.3, ramp_epochs=5)
        loader = DataLoader(dataset, sampler=sampler, batch_size=8)

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        dataset,
        scorer: SpanDifficultyScorer,
        start_pct: float = 0.30,
        ramp_epochs: int = 5,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.scorer = scorer
        self.start_pct = max(0.01, min(1.0, start_pct))
        self.ramp_epochs = max(1, ramp_epochs)
        self.seed = seed
        self._epoch = 0
        self._active_indices: List[int] = list(scorer.sorted_indices)  # start full; set_epoch updates

    def set_epoch(self, epoch: int) -> None:
        """Update active fraction based on current epoch. Call before each training epoch."""
        self._epoch = epoch
        t = min(1.0, epoch / self.ramp_epochs)
        fraction = self.start_pct + (1.0 - self.start_pct) * t
        n_active = max(1, int(fraction * len(self.dataset)))
        # Take the n_active easiest examples
        self._active_indices = self.scorer.sorted_indices[:n_active]

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        indices = list(self._active_indices)
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self._active_indices)
