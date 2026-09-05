"""Shared interface for graph-based GLiNER inference runtimes."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Mapping, Iterable

import numpy as np
import torch

from ..modeling.outputs import GLiNERBaseOutput, GLiNERRelexOutput


class BaseRuntimeModel(ABC):
    """Backend-independent adapter for exported GLiNER models.

    Concrete adapters only need to create and introspect their backend session,
    then implement :meth:`run_inference`. Input binding and GLiNER output
    construction are shared because those semantics are encoded in the graph's
    input and output names.
    """

    runtime_name = "runtime"

    def __init__(
        self,
        session: Any,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> None:
        self.session = session
        self.input_names = self._index_names(input_names, kind="input")
        self.output_names = self._index_names(output_names, kind="output")

    @staticmethod
    def _index_names(names: Iterable[str], *, kind: str) -> dict[str, int]:
        ordered_names = [str(name) for name in names]
        if not ordered_names:
            raise ValueError(f"The runtime model does not expose any {kind} names.")
        indexed = {name: index for index, name in enumerate(ordered_names)}
        if len(indexed) != len(ordered_names):
            raise ValueError(f"The runtime model exposes duplicate {kind} names.")
        return indexed

    @property
    def device(self) -> torch.device:
        """Return the host device used to stage runtime inputs."""
        return torch.device("cpu")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def prepare_inputs(self, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Convert values to NumPy, filter extras, and validate graph inputs."""
        if not isinstance(inputs, Mapping):
            raise ValueError("Inputs must be a mapping of input names to values.")

        unexpected = [name for name in inputs if name not in self.input_names]
        if unexpected:
            unexpected_names = ", ".join(sorted(unexpected))
            warnings.warn(
                f"Inputs not present in the runtime graph were ignored: {unexpected_names}.",
                stacklevel=2,
            )

        prepared = {
            name: self._to_numpy(inputs[name])
            for name in self.input_names
            if name in inputs and inputs[name] is not None
        }
        missing = [name for name in self.input_names if name not in prepared]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(f"Missing required runtime model inputs: {missing_names}.")
        return prepared

    def _bind_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        input_order = list(self.input_names)
        if len(args) == 1 and isinstance(args[0], Mapping):
            bound = dict(args[0])
        else:
            if len(args) > len(input_order):
                raise TypeError(f"Expected at most {len(input_order)} positional inputs, received {len(args)}.")
            bound = dict(zip(input_order, args, strict=False))

        for name in self.input_names:
            if name not in kwargs:
                continue
            if name in bound:
                raise TypeError(f"Runtime input '{name}' was provided more than once.")
            bound[name] = kwargs[name]
        return bound

    @staticmethod
    def _wrap_outputs(outputs: Mapping[str, Any]) -> GLiNERBaseOutput:
        if "logits" not in outputs:
            available = ", ".join(outputs) or "none"
            raise ValueError(f"Runtime graph must expose a 'logits' output; found: {available}.")

        relation_names = {"rel_idx", "rel_logits", "rel_mask"}
        present_relation_names = relation_names.intersection(outputs)
        if present_relation_names and present_relation_names != relation_names:
            missing = ", ".join(sorted(relation_names - present_relation_names))
            raise ValueError(f"Runtime relation outputs are incomplete; missing: {missing}.")

        is_relex = bool(present_relation_names)
        output_class = GLiNERRelexOutput if is_relex else GLiNERBaseOutput
        supported_names = output_class.__dataclass_fields__
        values = {name: value for name, value in outputs.items() if name in supported_names}
        return output_class(**values)

    def forward(self, *args: Any, **kwargs: Any) -> GLiNERBaseOutput:
        """Bind graph inputs, execute the backend, and return a GLiNER output."""
        inputs = self._bind_inputs(args, kwargs)
        prepared_inputs = self.prepare_inputs(inputs)
        return self._wrap_outputs(self.run_inference(prepared_inputs))

    def __call__(self, *args: Any, **kwargs: Any) -> GLiNERBaseOutput:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def run_inference(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute a prepared input mapping and return outputs keyed by name."""
