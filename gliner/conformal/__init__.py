"""Conformal-prediction coverage/risk guarantees for GLiNER zero-shot NER.

See docs/research/design.md for the full design rationale, and
docs/conformal.md (once written) for the practitioner-facing guide.
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
