__version__ = "0.2.28"

from .model import GLiNER
from .config import GLiNERConfig, StreamingSpanConfig
from .streaming import StreamingBatch, AsyncStreamingEngine
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
    "AsyncStreamingEngine",
    "GLiNER",
    "GLiNERConfig",
    "InferencePackingConfig",
    "PackedBatch",
    "StreamingBatch",
    "StreamingSpanConfig",
    "pack_requests",
    "unpack_spans",
]
