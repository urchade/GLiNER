"""ConformalGLiNER -- conformal-prediction wrapper around a span-mode GLiNER model.

See docs/research/design.md for the full design rationale. Summary of the one
behavior every method below enforces (design.md §0/§5): the ``>= 1-alpha``
guarantee applies only to entity types adequately represented in the
calibration set (``>= calibration_floor(alpha)`` gold occurrences). Any other
type is served from GLiNER's original uncalibrated ``p > 0.5`` rule, flagged
``"calibrated": False``, with a loud warning -- never silently blended into a
guaranteed-looking number.
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Union, Optional, Sequence
from collections import Counter, defaultdict
from dataclasses import field, dataclass

import torch

from gliner.decoding.decoder import Span

from .scores import align_gold_scores, extract_raw_scores
from .calibrators import calibration_floor, crc_lambda_search, mondrian_calibrate, split_conformal_quantile

_VALID_MODES = {"span_filter", "risk_control", "mondrian"}


@dataclass
class _CalibrationState:
    mode: str
    alpha: float
    labels: List[str]
    calibrated_types: List[str]
    type_counts: Dict[str, int]
    pooled_nc_threshold: Optional[float] = None
    mondrian_thresholds: Dict[str, float] = field(default_factory=dict)
    mondrian_skipped: Dict[str, int] = field(default_factory=dict)
    crc_lambda: Optional[float] = None
    model_id: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "labels": self.labels,
            "calibrated_types": self.calibrated_types,
            "type_counts": self.type_counts,
            "pooled_nc_threshold": self.pooled_nc_threshold,
            "mondrian_thresholds": self.mondrian_thresholds,
            "mondrian_skipped": self.mondrian_skipped,
            "crc_lambda": self.crc_lambda,
            "model_id": self.model_id,
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> _CalibrationState:
        return cls(**d)


class ConformalGLiNER:
    """Wraps a span-mode GLiNER model with a calibrated conformal filter.

    Never mutates the wrapped model. See docs/research/design.md §3 for the API
    rationale and §0 for exactly what the guarantee does and does not cover.
    """

    def __init__(self, model: Any):
        self.model = model
        self._state: Optional[_CalibrationState] = None

    @property
    def is_calibrated(self) -> bool:
        return self._state is not None

    def _require_calibrated(self) -> _CalibrationState:
        if self._state is None:
            raise RuntimeError("ConformalGLiNER is not calibrated. Call calibrate() first.")
        return self._state

    @staticmethod
    def _model_id(model: Any) -> Optional[str]:
        return getattr(getattr(model, "config", None), "_name_or_path", None)

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        calib_data: Sequence[Dict[str, Any]],
        alpha: float,
        mode: str = "risk_control",
        labels: Optional[Sequence[str]] = None,
    ) -> ConformalGLiNER:
        """Calibrate the conformal threshold(s) on a held-out labeled set.

        Args:
            calib_data: ``[{"tokenized_text": [...], "ner": [[start,end,type],...]}, ...]``
                -- the same schema GLiNER's own training/eval pipeline uses
                (gliner/data_processing/processor.py). Must be disjoint from any
                data later passed to :meth:`coverage_report` (design.md §"Split
                strategy" / eval_plan.md §2.2) -- reusing calibration examples to
                also measure coverage produces a biased, inflated estimate.
            alpha: target miscoverage/risk level in (0, 1).
            mode: one of ``"span_filter"``, ``"risk_control"``, ``"mondrian"``
                (design.md §1). No default is silently assumed by the public
                API surface beyond this parameter's own default; callers relying
                on the default should be aware it is ``"risk_control"``.
            labels: the fixed target label set 𝒯_cal. Defaults to every type
                appearing at least once in ``calib_data``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: invalid ``mode``/``alpha``, or too few calibration
                examples for the requested ``alpha`` (design.md §6 -- raises
                rather than silently degrading).
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not calib_data:
            raise ValueError("calib_data is empty.")

        if labels is None:
            labels = sorted({etype for ex in calib_data for (_, _, etype) in ex.get("ner", [])})
        labels = list(labels)
        if not labels:
            raise ValueError("No labels found in calib_data and none provided explicitly.")

        raw = extract_raw_scores(self.model, calib_data, labels)
        scores, types, example_idx = align_gold_scores(raw, calib_data)

        floor = calibration_floor(alpha)
        type_counts = Counter(types)
        calibrated_types = sorted(t for t, n in type_counts.items() if n >= floor)
        if not calibrated_types:
            raise ValueError(
                f"No requested type reached the calibration floor (>= {floor} gold occurrences "
                f"needed for alpha={alpha}). Observed counts: {dict(type_counts)}. Collect more "
                "calibration data, request fewer/more-common types, or use a larger alpha."
            )
        under_floor = {t: n for t, n in type_counts.items() if n < floor}
        if under_floor:
            warnings.warn(
                f"Type(s) {under_floor} have fewer than {floor} gold calibration occurrences "
                f"(alpha={alpha}) and will NOT receive a calibrated guarantee at predict time "
                "(raw uncalibrated p>0.5 fallback will be used for them, flagged accordingly).",
                UserWarning,
                stacklevel=2,
            )

        state = _CalibrationState(
            mode=mode,
            alpha=alpha,
            labels=labels,
            calibrated_types=calibrated_types,
            type_counts=dict(type_counts),
            model_id=self._model_id(self.model),
        )

        if mode in ("span_filter", "mondrian"):
            # Pooled threshold: theory.md part (iii-a), the marginal-over-calibrated-types
            # guarantee, and (for mondrian) the fallback for any calibrated-but-not-enough-
            # for-its-own-Mondrian-cell type -- though by construction every type in
            # `calibrated_types` already met the same floor, so mondrian_calibrate below
            # should not skip any of them; the pooled value is kept regardless as the
            # documented, deterministic fallback path (design.md §1.3).
            pooled_scores = [s for s, t in zip(scores, types) if t in calibrated_types]
            state.pooled_nc_threshold = split_conformal_quantile(pooled_scores, alpha)

        if mode == "mondrian":
            scores_by_type: Dict[str, List[float]] = defaultdict(list)
            for s, t in zip(scores, types):
                if t in calibrated_types:
                    scores_by_type[t].append(s)
            state.mondrian_thresholds, state.mondrian_skipped = mondrian_calibrate(scores_by_type, alpha)

        if mode == "risk_control":
            gold_nc_scores: List[List[float]] = [[] for _ in calib_data]
            for s, t, i in zip(scores, types, example_idx):
                if t in calibrated_types:
                    gold_nc_scores[i].append(s)
            state.crc_lambda = crc_lambda_search(gold_nc_scores, alpha)

        self._state = state
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _nc_threshold_for(self, state: _CalibrationState, etype: str) -> float:
        if state.mode == "risk_control":
            return state.crc_lambda
        if state.mode == "span_filter":
            return state.pooled_nc_threshold
        if state.mode == "mondrian":
            return state.mondrian_thresholds.get(etype, state.pooled_nc_threshold)
        raise AssertionError(f"unreachable mode {state.mode!r}")

    def predict_entities(
        self,
        text: Union[str, List[str]],
        labels: Sequence[str],
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Predict entities with conformal-guarantee metadata attached.

        Behaves like ``GLiNER.predict_entities``/``batch_predict_entities`` in
        shape (single text -> flat list; list of texts -> list of lists), but
        the admission rule is the calibrated conformal threshold, not a raw 0.5
        cutoff, for every type in ``labels`` that was adequately represented at
        calibration time. Every returned entity carries a
        ``"conformal": {"mode", "alpha", "calibrated"}`` field;
        ``"calibrated": False`` means that entity's type had no valid
        guarantee and was produced by the original uncalibrated rule instead
        (design.md §5).
        """
        state = self._require_calibrated()
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        labels = list(labels)

        prepared = self.model.prepare_batch(texts, labels)
        if not prepared["valid_texts"]:
            empty: List[List[Dict[str, Any]]] = [[] for _ in texts]
            return empty[0] if single else empty

        batch = self.model.collate_batch(prepared["input_x"], prepared["entity_types"])
        model_output = self.model.run_batch(batch, threshold=0.0, move_to_device=True)
        logits = model_output.logits if hasattr(model_output, "logits") else model_output[0]
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        probs = torch.sigmoid(logits)
        B, _, _, C = probs.shape

        id_to_classes = batch["id_to_classes"]
        if not isinstance(id_to_classes, list):
            id_to_classes = [id_to_classes] * B

        num_tokens = [len(t) for t in batch["tokens"]]

        uncalibrated_requested: set = set()
        decoded_per_item: List[List[Any]] = []

        for b in range(B):
            cls_map = id_to_classes[b]
            spans: List[Span] = []
            for col in range(C):
                etype = cls_map.get(col + 1)
                if etype is None:
                    continue
                calibrated = etype in state.calibrated_types
                if not calibrated:
                    uncalibrated_requested.add(etype)
                    admit_col = probs[b, :, :, col] > 0.5
                else:
                    tau = self._nc_threshold_for(state, etype)
                    admit_col = (1.0 - probs[b, :, :, col]) <= tau
                s_idx, k_idx = torch.where(admit_col)
                for s, k in zip(s_idx.tolist(), k_idx.tolist()):
                    if s + k >= num_tokens[b]:
                        continue
                    score = probs[b, s, k, col].item()
                    spans.append(Span(start=s, end=s + k, entity_type=etype, score=score))
            decoded_per_item.append(self.model.decoder.greedy_search(spans, flat_ner=flat_ner, multi_label=multi_label))

        if uncalibrated_requested:
            warnings.warn(
                f"Type(s) {sorted(uncalibrated_requested)} were not adequately represented in "
                f"calibration and have NO conformal guarantee -- served via GLiNER's original "
                "uncalibrated p>0.5 rule instead. Entities of these types are flagged "
                '"conformal": {"calibrated": False} in the output.',
                UserWarning,
                stacklevel=2,
            )

        entities = self.model.map_entities_to_text(
            decoded_per_item,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )
        for per_text in entities:
            for ent in per_text:
                calibrated = ent["label"] in state.calibrated_types
                ent["conformal"] = {"mode": state.mode, "alpha": state.alpha, "calibrated": calibrated}

        return entities[0] if single else entities

    # ------------------------------------------------------------------ #
    # Empirical validation
    # ------------------------------------------------------------------ #

    def coverage_report(
        self, test_data: Sequence[Dict[str, Any]], labels: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Empirically measure coverage/efficiency on held-out labeled data.

        ``test_data`` must be disjoint from whatever was passed to
        :meth:`calibrate` -- reusing calibration data here trivially inflates
        the coverage estimate (design.md §"Split strategy"; eval_plan.md §2.2).
        This method does not enforce disjointness itself (it has no way to know
        the calibration set's identity at this layer); callers/tests are
        responsible, per eval_plan.md's recommended "canary" regression test.

        Returns a dict with overall + per-type coverage (design.md/eval_plan.md
        §3.1/§3.3, restricted to calibrated types -- never blended with
        uncalibrated ones, design.md §5 point 3) and efficiency (§3.2).

        ``overall_coverage`` reports the quantity actually calibrated for
        ``state.mode``, not a one-size-fits-all pooled statistic: for
        ``"span_filter"``/``"mondrian"`` that's the marginal per-entity coverage
        (pooled over every gold entity, theory.md iii-a/iii-c); for
        ``"risk_control"`` it's ``1 - mean_per_sentence_miss_rate``, matching
        CRC's own loss definition (theory.md Eq. 4) exactly. These are genuinely
        different quantities whenever gold-entity count varies across sentences
        (theory.md part ii's "informative m" point) -- pooling entities flat for
        risk_control would silently report an uncalibrated number and can show
        spurious undercoverage unrelated to whether the actual CRC guarantee
        holds. (Caught empirically while validating this module -- see
        docs/research/validation_results.md.)
        """
        state = self._require_calibrated()
        labels = list(labels) if labels else list(state.labels)

        raw = extract_raw_scores(self.model, test_data, labels)
        scores, types, example_idx = align_gold_scores(raw, test_data)

        per_type_hits: Dict[str, int] = defaultdict(int)
        per_type_n: Dict[str, int] = defaultdict(int)
        n_uncalibrated_gold = 0
        per_example_gold: Dict[int, List[bool]] = defaultdict(list)
        for s, t, ex_i in zip(scores, types, example_idx):
            if t not in state.calibrated_types:
                n_uncalibrated_gold += 1
                continue
            tau = self._nc_threshold_for(state, t)
            hit = s <= tau
            per_type_n[t] += 1
            per_type_hits[t] += int(hit)
            per_example_gold[ex_i].append(hit)

        total_n = sum(per_type_n.values())
        total_hits = sum(per_type_hits.values())

        if state.mode == "risk_control":
            sentence_losses = [
                1.0 - sum(hits) / len(hits) if hits else 0.0
                for hits in (per_example_gold.get(i, []) for i in range(len(test_data)))
            ]
            overall_coverage = 1.0 - sum(sentence_losses) / len(sentence_losses) if sentence_losses else float("nan")
        else:
            overall_coverage = (total_hits / total_n) if total_n else float("nan")

        # Efficiency: mean admitted (span,type) pairs per example, over the full dense
        # candidate grid (not just gold cells) -- reuses the same forward pass, no extra cost.
        probs = torch.sigmoid(raw.logits)
        B = probs.shape[0]
        admitted_counts = torch.zeros(B)
        raw_candidate_counts = torch.zeros(B)
        for b in range(B):
            cls_map = raw.id_to_classes[b]
            for col in range(probs.shape[3]):
                etype = cls_map.get(col + 1)
                if etype is None or etype not in state.calibrated_types:
                    continue
                tau = self._nc_threshold_for(state, etype)
                admitted_counts[b] += ((1.0 - probs[b, :, :, col]) <= tau).sum().item()
                raw_candidate_counts[b] += probs.shape[1] * probs.shape[2]

        return {
            "mode": state.mode,
            "alpha": state.alpha,
            "n_test_examples": len(test_data),
            "overall_coverage": overall_coverage,
            "n_calibrated_gold": total_n,
            "n_uncalibrated_gold": n_uncalibrated_gold,
            "per_type_coverage": {t: per_type_hits[t] / per_type_n[t] for t in per_type_n},
            "per_type_n": dict(per_type_n),
            "efficiency_mean": admitted_counts.mean().item(),
            "raw_candidates_mean": raw_candidate_counts.mean().item(),
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def save_calibration(self, path: str) -> None:
        """Serialize calibration state (not the model) to JSON."""
        state = self._require_calibrated()
        with open(path, "w") as f:
            json.dump(state.to_json_dict(), f, indent=2)

    @classmethod
    def load_calibration(cls, path: str, model: Any) -> ConformalGLiNER:
        """Re-wrap ``model`` with a previously saved calibration state.

        Warns (does not raise) if ``model``'s identity doesn't match the model
        the calibration was computed against -- nonconformity scores are
        model-specific, so a mismatch means the loaded thresholds may not carry
        a valid guarantee for this model, but a deliberate same-architecture
        swap (e.g. a re-exported checkpoint) is a legitimate use case.
        """
        with open(path) as f:
            d = json.load(f)
        state = _CalibrationState.from_json_dict(d)
        current_id = cls._model_id(model)
        if state.model_id is not None and current_id is not None and state.model_id != current_id:
            warnings.warn(
                f"Loaded calibration was computed against model {state.model_id!r}, but this "
                f"model is {current_id!r}. Nonconformity scores are model-specific -- the "
                "guarantee may not hold unless this is a deliberate, compatible swap.",
                UserWarning,
                stacklevel=2,
            )
        cg = cls(model)
        cg._state = state
        return cg
