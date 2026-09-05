"""OpenVINO adapter for exported GLiNER models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import BaseRuntimeModel

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    import openvino as ov


def _require_openvino() -> Any:
    try:
        return importlib.import_module("openvino")
    except ImportError as error:
        raise ImportError(
            "OpenVINO is required for OpenVINO export and runtime. Install it with `pip install gliner[openvino]`."
        ) from error


def _port_name(port: Any, *, kind: str, index: int) -> str:
    try:
        return port.get_any_name()
    except (AttributeError, RuntimeError):
        name = getattr(port, "any_name", None)
        if name:
            return str(name)
        raise ValueError(f"OpenVINO {kind} port {index} does not expose a tensor name.") from None


class OpenVINOModel(BaseRuntimeModel):
    """Run any supported exported GLiNER ONNX or IR graph with OpenVINO."""

    runtime_name = "openvino"

    def __init__(
        self,
        compiled_model: ov.CompiledModel | None = None,
        model_path: str | PathLike[str] | None = None,
        device_name: str = "CPU",
        config: dict[str, Any] | None = None,
        core: ov.Core | None = None,
    ) -> None:
        self.model_path = model_path
        self.device_name = device_name
        self.compile_config = {} if config is None else dict(config)
        self.core = core

        if compiled_model is None:
            if model_path is None:
                raise ValueError("Either 'compiled_model' or 'model_path' must be provided.")
            compiled_model = self._load_compiled_model(model_path)

        self.compiled_model = compiled_model
        self._input_ports = list(compiled_model.inputs)
        self._output_ports = list(compiled_model.outputs)
        super().__init__(
            compiled_model,
            (_port_name(port, kind="input", index=index) for index, port in enumerate(self._input_ports)),
            (_port_name(port, kind="output", index=index) for index, port in enumerate(self._output_ports)),
        )

    def _load_compiled_model(self, model_path: str | PathLike[str]) -> ov.CompiledModel:
        if self.core is None:
            runtime = _require_openvino()
            self.core = runtime.Core()
        model = self.core.read_model(str(model_path))
        return self.core.compile_model(model, device_name=self.device_name, config=self.compile_config)

    def run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # The callable interface accepts named NumPy inputs and returns an
        # OVDict whose output-port keys remain stable across inference calls.
        values = self.compiled_model(inputs)
        return {name: values[self._output_ports[index]] for name, index in self.output_names.items()}


__all__ = ["OpenVINOModel"]
