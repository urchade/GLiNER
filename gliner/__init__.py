__version__ = "0.2.27"

from .model import GLiNER
from .config import GLiNERConfig
from .descriptions import (
    normalise_labels,
    ONTONOTES_DESCRIPTIONS,
    CONLL_DESCRIPTIONS,
    WNUT_DESCRIPTIONS,
    BIOMEDICAL_DESCRIPTIONS,
)
from .infer_packing import (
    PackedBatch,
    InferencePackingConfig,
    unpack_spans,
    pack_requests,
)

# from .multitask import (GLiNERClassifier, GLiNERQuestionAnswerer, GLiNEROpenExtractor,
#                                 GLiNERRelationExtractor, GLiNERSummarizer, GLiNERSquadEvaluator,
#                                     GLiNERDocREDEvaluator)

__all__ = [
    "GLiNER",
    "GLiNERConfig",
    "InferencePackingConfig",
    "PackedBatch",
    "pack_requests",
    "unpack_spans",
    "normalise_labels",
    "ONTONOTES_DESCRIPTIONS",
    "CONLL_DESCRIPTIONS",
    "WNUT_DESCRIPTIONS",
    "BIOMEDICAL_DESCRIPTIONS",
]
