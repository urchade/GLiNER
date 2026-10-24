"""Counts of the batches a serve replica runs through the model."""

from __future__ import annotations

from typing import Any
from collections import Counter


class BatchStats:
    """Forward passes by batch size: the rows requests asked for, and the rows the GPU ran.

    The two differ when compilation pads a batch up to a precompiled batch size.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rows: Counter[int] = Counter()
        self.gpu_rows: Counter[int] = Counter()

    def record(self, rows: int, gpu_rows: int | None = None) -> None:
        self.rows[rows] += 1
        self.gpu_rows[rows if gpu_rows is None else gpu_rows] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_passes": sum(self.rows.values()),
            "rows": sum(size * count for size, count in self.rows.items()),
            "max_rows": max(self.rows, default=0),
            "rows_per_pass": {str(size): count for size, count in sorted(self.rows.items())},
            "gpu_rows_per_pass": {str(size): count for size, count in sorted(self.gpu_rows.items())},
        }
