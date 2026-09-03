"""ONNX Runtime adapter for exported GLiNER models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch

from .base import BaseRuntimeModel

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    import onnxruntime as ort


def _require_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as error:
        raise ImportError(
            "ONNX Runtime is required for runtime='onnxruntime'. Install it with "
            "`pip install onnxruntime` (or `onnxruntime-gpu` for GPU execution)."
        ) from error


class ONNXRuntimeModel(BaseRuntimeModel):
    """Run any supported exported GLiNER graph with ONNX Runtime."""

    runtime_name = "onnxruntime"

    def __init__(
        self,
        session: ort.InferenceSession | None = None,
        model_path: str | PathLike[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        providers: list[str] | None = None,
    ) -> None:
        if session is None:
            if model_path is None:
                raise ValueError("Either 'session' or 'model_path' must be provided.")
            runtime = _require_onnxruntime()
            if session_options is None:
                session_options = runtime.SessionOptions()
                session_options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = runtime.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )

        self.model_path = model_path
        self.session_options = session_options
        self.providers = providers
        super().__init__(
            session,
            (input_info.name for input_info in session.get_inputs()),
            (output_info.name for output_info in session.get_outputs()),
        )

    @property
    def device(self) -> torch.device:
        providers = None
        if hasattr(self.session, "get_providers"):
            providers = self.session.get_providers()
        if providers is None:
            providers = self.providers
        provider_names = [provider[0] if isinstance(provider, tuple) else provider for provider in providers or ()]
        if "CUDAExecutionProvider" in provider_names:
            return torch.device("cuda")
        return torch.device("cpu")

    def run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = self.session.run(None, inputs)
        return {name: values[index] for name, index in self.output_names.items()}


__all__ = ["ONNXRuntimeModel"]
