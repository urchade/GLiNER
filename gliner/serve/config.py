"""Configuration for GLiNER Ray Serve deployment."""

from typing import List, Optional
from dataclasses import field, dataclass


@dataclass
class GLiNERServeConfig:
    """Configuration for GLiNER Ray Serve deployment.

    This config controls model loading, serving parameters, and dynamic batching behavior.
    Aligned with GLiNEREngineConfig from engine module.
    """

    model: str
    device: str = "cuda"
    dtype: str = "bfloat16"

    quantization: Optional[str] = None

    max_model_len: int = 2048
    max_span_width: int = 12
    max_labels: int = -1

    default_threshold: float = 0.5
    default_relation_threshold: float = 0.5

    num_replicas: int = 1
    num_gpus_per_replica: float = 1.0
    num_cpus_per_replica: float = 1.0

    max_batch_size: int = 32
    batch_wait_timeout_ms: float = 5.0
    request_timeout_s: float = 30.0
    max_ongoing_requests: int = 256
    queue_capacity: int = 4096

    route_prefix: str = "/gliner"

    tokenizer_threads: int = 4
    decoding_threads: int = 4

    enable_compilation: bool = True
    enable_sequence_packing: bool = False
    enable_flashdeberta: bool = False

    precompiled_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    target_memory_fraction: float = 0.8
    memory_overhead_factor: float = 1.3

    calibration_min_seq_len: int = 64
    calibration_probe_batch_size: int = 2

    warmup_iterations: int = 3

    http_port: int = 8000

    ray_address: Optional[str] = None

    def __post_init__(self):
        if self.max_batch_size not in self.precompiled_batch_sizes:
            self.precompiled_batch_sizes = sorted(set(self.precompiled_batch_sizes) | {self.max_batch_size})
        self.precompiled_batch_sizes = sorted(self.precompiled_batch_sizes)

    def to_env_vars(self) -> dict:
        """Convert config to environment variables for model loading."""
        env = {}
        if self.enable_flashdeberta:
            env["USE_FLASHDEBERTA"] = "1"
        if self.tokenizer_threads > 0:
            env["TOKENIZERS_PARALLELISM"] = "true"
        return env
