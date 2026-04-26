"""GLiNER Ray Serve module for production deployment.

Quick Start:

    # Start server (CLI)
    python -m gliner.serve --model urchade/gliner_small-v2.1

    # Make predictions (Python)
    from gliner.serve import GLiNERClient
    client = GLiNERClient()
    result = client.predict("John works at Google", ["person", "organization"])

    # Or programmatically start server
    from gliner.serve import GLiNERServeConfig, serve
    config = GLiNERServeConfig(model="urchade/gliner_small-v2.1")
    handle = serve(config)

Features:
    - Dynamic batching via Ray Serve
    - Memory-aware batch sizing (prevents CUDA OOM)
    - Precompiled power-of-two batch sizes
    - NER and relation extraction support
    - FlashDeBERTa and sequence packing

See docs/serving.md for full documentation.
"""

from .client import GLiNERClient, get_client
from .config import GLiNERServeConfig
from .memory import GLiNERMemoryEstimator
from .server import GLiNERServer, GLiNERFactory, serve, shutdown

__all__ = [
    "GLiNERClient",
    "GLiNERFactory",
    "GLiNERMemoryEstimator",
    "GLiNERServeConfig",
    "GLiNERServer",
    "get_client",
    "serve",
    "shutdown",
]
