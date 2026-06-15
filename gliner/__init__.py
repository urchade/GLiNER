__version__ = "0.2.27"

from .model import GLiNER
from .config import GLiNERConfig
from .descriptions import (
    WNUT_DESCRIPTIONS,
    CONLL_DESCRIPTIONS,
    ONTONOTES_DESCRIPTIONS,
    BIOMEDICAL_DESCRIPTIONS,
    normalise_labels,
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
    "BIOMEDICAL_DESCRIPTIONS",
    "CONLL_DESCRIPTIONS",
    "ONTONOTES_DESCRIPTIONS",
    "WNUT_DESCRIPTIONS",
    "GLiNER",
    "GLiNERConfig",
    "InferencePackingConfig",
    "PackedBatch",
    "normalise_labels",
    "pack_requests",
    "unpack_spans",
]
