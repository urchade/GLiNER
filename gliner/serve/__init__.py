"""GLiNER Ray Serve module for production deployment.

Quick Start:

    # Start server (CLI)
    python -m gliner.serve --model urchade/gliner_small-v2.1

    # Make predictions (Python)
    from gliner.serve import GLiNERClient
    client = GLiNERClient()
    result = client.predict("John works at Google", ["person", "organization"])

    # Or programmatically start server
    from gliner.serve import GLiNERFactoryConfig, serve
    config = GLiNERFactoryConfig(model="urchade/gliner_small-v2.1")
    handle = serve(config)

Features:
    - Dynamic batching via Ray Serve
    - Memory-aware batch sizing (prevents CUDA OOM)
    - Precompiled power-of-two batch sizes
    - NER and relation extraction support
    - FlashDeBERTa and sequence packing

See docs/serving.md for full documentation.
"""

from .config import GLiNERFactoryConfig
from .memory import GLiNERMemoryEstimator
from .server import GLiNERFactory, GLiNERServer, serve, shutdown
from .client import GLiNERClient, get_client

__all__ = [
    "GLiNERFactoryConfig",
    "GLiNERMemoryEstimator",
    "GLiNERFactory",
    "GLiNERServer",
    "GLiNERClient",
    "serve",
    "shutdown",
    "get_client",
]
