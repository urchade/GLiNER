"""Memory estimation for GLiNER via precomputed calibration table.

Startup calibration runs the model on probe batches at power-of-two sequence
lengths and records peak GPU memory per sample. At request time ``batch_size_fn``
picks the largest precompiled batch size that satisfies

    per_sample(seq_len) * N  <=  total_gpu - cuda_context - model_weights

using a pessimistic (rounded-up) seq_len and a safety factor on per-sample
memory. Labels and relations are NOT scaled as a separate dimension — they are
part of the model input, so callers must include their word count in
``seq_len`` when invoking ``batch_size_fn``.
"""

import logging
from typing import Dict, List, Callable

import torch

logger = logging.getLogger(__name__)


def _power_of_two_seq_lens(max_seq_len: int, min_seq_len: int = 64) -> List[int]:
    """Return power-of-two sequence lengths from min_seq_len up to max_seq_len."""
    lens: List[int] = []
    s = max(1, min_seq_len)
    while s < max_seq_len:
        lens.append(s)
        s *= 2
    lens.append(max_seq_len)
    return lens


class GLiNERMemoryEstimator:
    """Precomputed memory table for GLiNER inference."""

    def __init__(
        self,
        safety_factor: float = 1.3,
        target_memory_fraction: float = 0.9,
        calibration_probe_batch_size: int = 2,
    ):
        self.safety_factor = safety_factor
        self.target_memory_fraction = target_memory_fraction
        self.calibration_probe_batch_size = max(2, calibration_probe_batch_size)

        self.total_gpu_memory: int = 0
        self.cuda_context_bytes: int = 0
        self.model_memory_bytes: int = 0

        self.per_sample_table: Dict[int, int] = {}

    def measure_cuda_context(self) -> None:
        """Record CUDA context overhead. Must be called before the model loads."""
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.total_gpu_memory = total
        self.cuda_context_bytes = total - free
        logger.info("CUDA context: %.1f MiB", self.cuda_context_bytes / (1024**2))

    def measure_model_memory(self) -> None:
        """Record model weight memory. Must be called after the model loads."""
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        self.total_gpu_memory = total
        used = total - free
        self.model_memory_bytes = max(0, used - self.cuda_context_bytes)
        logger.info("Model weights: %.1f MiB", self.model_memory_bytes / (1024**2))

    def available_memory(self) -> int:
        """Budget for a batch: ``total_gpu - cuda_context - model_weights``."""
        if not torch.cuda.is_available():
            return 0
        budget = self.total_gpu_memory - self.cuda_context_bytes - self.model_memory_bytes
        return max(0, int(budget * self.target_memory_fraction))

    def calibrate(
        self,
        batch_method: Callable,
        max_seq_len: int,
        min_seq_len: int = 64,
    ) -> None:
        """Populate ``per_sample_table`` across power-of-two seq lengths.

        Uses a single dummy label so the probed sequence length is dominated
        by text tokens; label/relation words are accounted for at lookup time
        by the caller extending ``seq_len``.
        """
        if not torch.cuda.is_available():
            return

        seq_lens = _power_of_two_seq_lens(max_seq_len, min_seq_len=min_seq_len)
        dummy_labels = ["label"]
        probe_b = self.calibration_probe_batch_size

        logger.info("Calibrating memory table: seq_lens=%s, probe_batch=%s", seq_lens, probe_b)

        for seq_len in seq_lens:
            dummy_text = "word " * max(1, seq_len // 2)
            peak = self._measure_peak(batch_method, [dummy_text] * probe_b, dummy_labels)
            per_sample = max(1, peak // probe_b)
            self.per_sample_table[seq_len] = per_sample
            logger.info("  seq_len=%5d: per_sample=%.1f MiB", seq_len, per_sample / (1024**2))

    def _measure_peak(
        self,
        batch_method: Callable,
        texts: List[str],
        labels: List[str],
    ) -> int:
        """Run a probe batch and return peak allocated bytes above baseline."""
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        batch_method(
            texts,
            labels,
            threshold=0.5,
            flat_ner=True,
            multi_label=False,
        )

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return max(0, peak - baseline)

    def _lookup_seq_len(self, seq_len: int) -> int:
        """Round ``seq_len`` up to the nearest calibrated entry (pessimistic)."""
        if not self.per_sample_table:
            raise RuntimeError("Memory estimator has not been calibrated")
        for key in sorted(self.per_sample_table.keys()):
            if key >= seq_len:
                return key
        return max(self.per_sample_table.keys())

    def per_sample_at(self, seq_len: int) -> int:
        """Pessimistic per-sample memory at or above ``seq_len``."""
        probe_seq_len = self._lookup_seq_len(seq_len)
        return int(self.per_sample_table[probe_seq_len] * self.safety_factor)

    def batch_size_fn(
        self,
        seq_len: int,
        precompiled_sizes: List[int],
    ) -> int:
        """Largest precompiled batch size satisfying ``per_sample * N <= budget``.

        Budget = ``total_gpu - cuda_context - model_weights`` (times the
        configured ``target_memory_fraction``). The caller is responsible for
        folding label / relation word counts into ``seq_len``.
        """
        if not precompiled_sizes:
            return 1

        available = self.available_memory()
        if available <= 0:
            return min(precompiled_sizes)

        per_sample = self.per_sample_at(seq_len)
        for size in sorted(precompiled_sizes, reverse=True):
            if per_sample * size <= available:
                return size
        return min(precompiled_sizes)
