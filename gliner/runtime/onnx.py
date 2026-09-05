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
            '`pip install "gliner[onnx]"` for CPU or `pip install "gliner[gpu]"` for GPU execution. '
            "Install only one ONNX Runtime package."
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
        self.model_path = model_path
        self.session_options = session_options
        self.providers = providers

        if session is None:
            if model_path is None:
                raise ValueError("Either session or model_path must be provided.")
            session = self._load_session(model_path)

        super().__init__(
            session,
            (input_info.name for input_info in session.get_inputs()),
            (output_info.name for output_info in session.get_outputs()),
        )

    def _load_session(self, model_path: str | PathLike[str]) -> ort.InferenceSession:
        runtime = _require_onnxruntime()
        if self.session_options is None:
            self.session_options = runtime.SessionOptions()
            self.session_options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        return runtime.InferenceSession(
            str(model_path),
            sess_options=self.session_options,
            providers=self.providers,
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
