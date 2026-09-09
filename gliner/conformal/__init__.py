"""Conformal-prediction coverage/risk guarantees for GLiNER zero-shot NER.

See docs/conformal.md for the practitioner-facing guide, including the design
rationale and known limitations.
"""

from .scores import RawScoreBatch, align_gold_scores, extract_raw_scores
from .wrapper import ConformalGLiNER
from .calibrators import calibration_floor, crc_lambda_search, mondrian_calibrate, split_conformal_quantile

__all__ = [
    "ConformalGLiNER",
    "RawScoreBatch",
    "align_gold_scores",
    "calibration_floor",
    "crc_lambda_search",
    "extract_raw_scores",
    "mondrian_calibrate",
    "split_conformal_quantile",
]
