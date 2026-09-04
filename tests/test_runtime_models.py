from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

import gliner.runtime.onnx as onnx_runtime
import gliner.runtime.openvino as openvino_runtime
from gliner.model import BaseGLiNER
from gliner.runtime import OpenVINOModel, BaseRuntimeModel, ONNXRuntimeModel
from gliner.modeling.outputs import GLiNERBaseOutput, GLiNERRelexOutput


class _RuntimeModel(BaseRuntimeModel):
    runtime_name = "fake"

    def __init__(self, outputs, input_names=("input_ids", "attention_mask")):
        super().__init__(
            session=object(),
            input_names=input_names,
            output_names=outputs,
        )
        self.outputs = outputs
        self.received_inputs = None

    def run_inference(self, inputs):
        self.received_inputs = inputs
        return self.outputs


class _ORTValueInfo:
    def __init__(self, name):
        self.name = name


class _ORTSession:
    def __init__(self, input_names, outputs, providers=None):
        self._inputs = [_ORTValueInfo(name) for name in input_names]
        self._outputs = [_ORTValueInfo(name) for name, _ in outputs]
        self._output_values = [value for _, value in outputs]
        self._providers = ["CPUExecutionProvider"] if providers is None else providers
        self.run_args = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def run(self, requested_outputs, inputs):
        self.run_args = (requested_outputs, inputs)
        return self._output_values


class _OpenVINOPort:
    def __init__(self, name):
        self._name = name

    def get_any_name(self):
        return self._name

    def get_tensor(self):
        return self

    def set_names(self, names):
        self._name = next(iter(names))


class _OpenVINOCompiledModel:
    def __init__(self, input_names, outputs):
        self.inputs = [_OpenVINOPort(name) for name in input_names]
        self.outputs = [_OpenVINOPort(name) for name, _ in outputs]
        self._results = {port: value for port, (_, value) in zip(self.outputs, outputs, strict=True)}
        self.received_inputs = None

    def __call__(self, inputs):
        self.received_inputs = inputs
        return self._results


class _OpenVINOCore:
    def __init__(self, compiled_model):
        self.compiled_model = compiled_model
        self.read_model_arg = None
        self.compile_args = None

    def read_model(self, model_path):
        self.read_model_arg = model_path
        return model_path

    def compile_model(self, model, device_name, config):
        self.compile_args = (model, device_name, config)
        return self.compiled_model


class _OpenVINOExportAPI:
    PartialShape = list

    def __init__(self, fail_conversion=False):
        self.fail_conversion = fail_conversion
        self.convert_args = None
        self.converted_model = None
        self.save_args = None

    def convert_model(self, model, example_input, input):
        self.convert_args = (model, example_input, input)
        if self.fail_conversion:
            raise RuntimeError("conversion failed")
        self.converted_model = type(
            "ConvertedModel",
            (),
            {
                "inputs": [_OpenVINOPort("") for _ in example_input],
                "outputs": [_OpenVINOPort("")],
            },
        )()
        return self.converted_model

    def save_model(self, model, output_path, compress_to_fp16):
        output_path = Path(output_path)
        self.save_args = (model, output_path, compress_to_fp16)
        output_path.write_text("xml", encoding="utf-8")
        output_path.with_suffix(".bin").write_bytes(b"weights")


class _ExportableModel:
    export_to_openvino = BaseGLiNER.export_to_openvino

    def __init__(self):
        self.export_args = None
        self.wrapper = object()
        self.example_inputs = (
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
        )
        self.spec = {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        }
        self.config = type(
            "ExportConfig",
            (),
            {"to_json_file": lambda _, path: Path(path).write_text("{}", encoding="utf-8")},
        )()
        tokenizer = type(
            "ExportTokenizer",
            (),
            {"save_pretrained": lambda _, path: (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")},
        )()
        self.data_processor = type("ExportProcessor", (), {"transformer_tokenizer": tokenizer})()

    def _check_export_preconditions(self):
        return None

    def _prepare_export_graph(self, **export_kwargs):
        self.export_args = export_kwargs
        return self.wrapper, self.example_inputs, self.spec


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Decoder:
    def __init__(self, config):
        self.config = config


class _LoaderGLiNER(BaseGLiNER):
    config_class = _Config
    decoder_class = _Decoder

    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        raise AssertionError("Runtime loading must inject a model adapter.")

    def _create_data_processor(self, config, cache_dir, tokenizer=None, **kwargs):
        return object()

    def resize_embeddings(self):
        return None

    def inference(self):
        return None

    def evaluate(self):
        return None


def test_base_runtime_prepares_expected_inputs_and_ignores_extras():
    model = _RuntimeModel({"logits": np.array([1.0], dtype=np.float32)})
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = np.array([[1, 1]], dtype=np.int64)

    with pytest.warns(UserWarning):
        prepared = model.prepare_inputs(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "not_in_graph": torch.tensor([99]),
            }
        )

    assert set(prepared) == {"input_ids", "attention_mask"}
    np.testing.assert_array_equal(prepared["input_ids"], input_ids.numpy())
    np.testing.assert_array_equal(prepared["attention_mask"], attention_mask)


def test_base_runtime_rejects_missing_graph_inputs():
    model = _RuntimeModel({"logits": np.array([1.0], dtype=np.float32)})

    with pytest.raises(ValueError, match="attention_mask"):
        model(input_ids=torch.tensor([[1, 2]]))


def test_base_runtime_maps_positional_inputs_and_builds_base_output():
    logits = np.array([[[0.25, 0.75]]], dtype=np.float32)
    model = _RuntimeModel({"logits": logits})

    output = model(
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[1, 1]], dtype=torch.long),
    )

    assert isinstance(output, GLiNERBaseOutput)
    assert not isinstance(output, GLiNERRelexOutput)
    assert output.logits is logits
    assert set(model.received_inputs) == {"input_ids", "attention_mask"}


def test_base_runtime_builds_relation_output_from_output_schema():
    outputs = {
        "logits": np.array([1.0], dtype=np.float32),
        "rel_idx": np.array([[0, 1]], dtype=np.int64),
        "rel_logits": np.array([0.8], dtype=np.float32),
        "rel_mask": np.array([1], dtype=np.int64),
    }
    model = _RuntimeModel(outputs)

    output = model(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
    )

    assert isinstance(output, GLiNERRelexOutput)
    assert output.logits is outputs["logits"]
    assert output.rel_idx is outputs["rel_idx"]
    assert output.rel_logits is outputs["rel_logits"]
    assert output.rel_mask is outputs["rel_mask"]


def test_base_runtime_rejects_incomplete_relation_output_schema():
    model = _RuntimeModel(
        {
            "logits": np.array([1.0], dtype=np.float32),
            "rel_logits": np.array([0.8], dtype=np.float32),
        }
    )

    with pytest.raises(ValueError, match="relation outputs are incomplete"):
        model(
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.tensor([[1]], dtype=torch.long),
        )


def test_onnx_runtime_discovers_names_and_maps_session_outputs():
    first = np.array([10], dtype=np.int64)
    logits = np.array([0.5], dtype=np.float32)
    session = _ORTSession(
        input_names=["attention_mask", "input_ids"],
        outputs=[("first", first), ("logits", logits)],
    )
    model = ONNXRuntimeModel(session=session)
    prepared = {
        "attention_mask": np.array([[1]], dtype=np.int64),
        "input_ids": np.array([[42]], dtype=np.int64),
    }

    outputs = model.run_inference(prepared)

    assert model.runtime_name == "onnxruntime"
    assert model.session is session
    assert model.input_names == {"attention_mask": 0, "input_ids": 1}
    assert model.output_names == {"first": 0, "logits": 1}
    assert session.run_args == (None, prepared)
    assert outputs["first"] is first
    assert outputs["logits"] is logits


def test_onnx_runtime_uses_effective_session_providers_for_device():
    session = _ORTSession(
        input_names=["input_ids"],
        outputs=[("logits", np.array([0.5], dtype=np.float32))],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    model = ONNXRuntimeModel(session=session, providers=["CPUExecutionProvider"])

    assert model.device == torch.device("cuda")


def test_openvino_runtime_maps_results_by_output_port():
    logits = np.array([0.5], dtype=np.float32)
    auxiliary = np.array([7], dtype=np.int64)
    compiled_model = _OpenVINOCompiledModel(
        input_names=["input_ids", "attention_mask"],
        outputs=[("logits", logits), ("auxiliary", auxiliary)],
    )
    model = OpenVINOModel(compiled_model=compiled_model)
    prepared = {
        "input_ids": np.array([[42]], dtype=np.int64),
        "attention_mask": np.array([[1]], dtype=np.int64),
    }

    outputs = model.run_inference(prepared)

    assert model.runtime_name == "openvino"
    assert model.session is compiled_model
    assert model.input_names == {"input_ids": 0, "attention_mask": 1}
    assert model.output_names == {"logits": 0, "auxiliary": 1}
    assert compiled_model.received_inputs is prepared
    assert outputs["logits"] is logits
    assert outputs["auxiliary"] is auxiliary


def test_openvino_runtime_passes_device_and_config_to_injected_core():
    compiled_model = _OpenVINOCompiledModel(
        input_names=["input_ids"],
        outputs=[("logits", np.array([0.5], dtype=np.float32))],
    )
    core = _OpenVINOCore(compiled_model)
    config = {"PERFORMANCE_HINT": "LATENCY"}

    model = OpenVINOModel(
        model_path="model.onnx",
        device_name="AUTO",
        config=config,
        core=core,
    )

    assert model.session is compiled_model
    assert core.read_model_arg == "model.onnx"
    assert core.compile_args == ("model.onnx", "AUTO", config)


def test_base_gliner_exports_openvino_ir_directly(monkeypatch, tmp_path):
    api = _OpenVINOExportAPI()
    monkeypatch.setattr(openvino_runtime, "_require_openvino", lambda: api)
    model = _ExportableModel()

    paths = model.export_to_openvino(
        tmp_path,
        "nested/gliner.xml",
        compress_to_fp16=True,
        labels=["person"],
    )

    xml_path = tmp_path / "nested" / "gliner.xml"
    assert paths == {
        "openvino_path": str(xml_path),
        "weights_path": str(xml_path.with_suffix(".bin")),
    }
    assert model.export_args == {"labels": ["person"]}
    assert api.convert_args == (
        model.wrapper,
        model.example_inputs,
        [[-1, -1], [-1, 4]],
    )
    assert [port.get_any_name() for port in api.converted_model.inputs] == ["input_ids", "attention_mask"]
    assert [port.get_any_name() for port in api.converted_model.outputs] == ["logits"]
    assert api.save_args[1:] == (xml_path, True)
    assert xml_path.is_file()
    assert xml_path.with_suffix(".bin").is_file()
    assert (tmp_path / "gliner_config.json").is_file()
    assert (tmp_path / "tokenizer.json").is_file()


def test_base_gliner_openvino_export_does_not_write_assets_after_conversion_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(openvino_runtime, "_require_openvino", lambda: _OpenVINOExportAPI(fail_conversion=True))
    model = _ExportableModel()

    with pytest.raises(RuntimeError, match="conversion failed"):
        model.export_to_openvino(tmp_path)

    assert not (tmp_path / "model.xml").exists()
    assert not (tmp_path / "gliner_config.json").exists()
    assert not (tmp_path / "tokenizer.json").exists()


def test_base_gliner_openvino_export_validates_filename_before_loading_dependency(monkeypatch, tmp_path):
    def unexpected_dependency_load():
        raise AssertionError("OpenVINO must not be loaded for an invalid output name")

    monkeypatch.setattr(openvino_runtime, "_require_openvino", unexpected_dependency_load)

    with pytest.raises(ValueError, match=r"\.xml extension"):
        _ExportableModel().export_to_openvino(tmp_path, "model.onnx")


def test_base_gliner_direct_openvino_export_supports_variadic_dynamic_graph(tmp_path):
    pytest.importorskip("openvino")

    class VariadicAdd(torch.nn.Module):
        def forward(self, *inputs):
            return inputs[0] + inputs[1]

    model = _ExportableModel()
    model.wrapper = VariadicAdd()
    model.spec["dynamic_axes"]["attention_mask"][1] = "sequence_length"
    paths = model.export_to_openvino(tmp_path)

    runtime = OpenVINOModel(model_path=paths["openvino_path"])
    for width in (4, 7):
        values = np.ones((1, width), dtype=np.int64)
        output = runtime(input_ids=values, attention_mask=values)
        np.testing.assert_array_equal(output.logits, values * 2)


@pytest.mark.parametrize(
    ("runtime", "runtime_options", "expected_class"),
    [
        (
            "onnxruntime",
            {"session": _ORTSession(["input_ids"], [("logits", np.array([0.5], dtype=np.float32))])},
            ONNXRuntimeModel,
        ),
        (
            "openvino",
            {
                "compiled_model": _OpenVINOCompiledModel(
                    ["input_ids"],
                    [("logits", np.array([0.5], dtype=np.float32))],
                )
            },
            OpenVINOModel,
        ),
    ],
)
def test_base_gliner_loader_wires_runtime_adapters_without_artifact_file(
    tmp_path,
    runtime,
    runtime_options,
    expected_class,
):
    (tmp_path / "gliner_config.json").write_text("{}", encoding="utf-8")

    model = _LoaderGLiNER.from_pretrained(
        str(tmp_path),
        model_dir=tmp_path,
        load_tokenizer=False,
        runtime=runtime,
        runtime_options=runtime_options,
    )

    assert isinstance(model.model, expected_class)
    assert model.runtime == runtime
    assert model.is_runtime_model
    assert model.onnx_model is (runtime == "onnxruntime")

    with pytest.raises(RuntimeError, match=r"torch\.compile is only available"):
        model.compile()


def test_openvino_runtime_loads_and_executes_real_ir(tmp_path):
    ov = pytest.importorskip("openvino")
    input_node = ov.opset13.parameter([1, 2], np.float32, name="input_ids")
    logits = ov.opset13.multiply(input_node, ov.opset13.constant(np.array(2.0, dtype=np.float32)))
    logits.output(0).get_tensor().set_names({"logits"})
    graph = ov.Model([logits], [input_node], "tiny_gliner_runtime")
    model_path = tmp_path / "model.xml"
    ov.save_model(graph, model_path, compress_to_fp16=False)

    model = OpenVINOModel(model_path=model_path, device_name="CPU")
    output = model(input_ids=np.array([[1.0, 3.0]], dtype=np.float32))

    assert isinstance(output, GLiNERBaseOutput)
    np.testing.assert_allclose(output.logits, np.array([[2.0, 6.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("module", "model_class", "dependency_name"),
    [
        (onnx_runtime, ONNXRuntimeModel, "onnxruntime"),
        (openvino_runtime, OpenVINOModel, "openvino"),
    ],
)
def test_runtime_loaders_report_missing_optional_dependency(monkeypatch, module, model_class, dependency_name):
    def missing_dependency(name):
        assert name == dependency_name
        raise ImportError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing_dependency)

    with pytest.raises(ImportError, match=dependency_name):
        model_class(model_path="model.onnx")


@pytest.mark.parametrize(
    ("runtime", "load_onnx_model", "expected"),
    [
        (None, False, "torch"),
        (None, True, "onnxruntime"),
        ("pytorch", False, "torch"),
        ("onnx", False, "onnxruntime"),
        ("ORT", False, "onnxruntime"),
        ("ov", False, "openvino"),
        ("OpenVINO", False, "openvino"),
    ],
)
def test_base_gliner_normalizes_runtime_aliases(runtime, load_onnx_model, expected):
    assert BaseGLiNER._normalize_runtime(runtime, load_onnx_model) == expected


def test_base_gliner_rejects_conflicting_legacy_runtime_flag():
    with pytest.raises(ValueError, match="load_onnx_model=True conflicts"):
        BaseGLiNER._normalize_runtime("openvino", load_onnx_model=True)


def test_base_gliner_rejects_torch_only_options_for_external_runtime():
    with pytest.raises(ValueError, match="dtype, quantize only apply to the PyTorch runtime"):
        BaseGLiNER._validate_runtime_only_options(
            "openvino",
            variant=None,
            dtype="fp16",
            quantize="int8",
            compile_torch_model=False,
            low_cpu_mem_usage=False,
        )


@pytest.mark.parametrize("runtime", ["onnxruntime", "openvino"])
@pytest.mark.parametrize(
    "config",
    [
        _Config(model_type="gliner_streaming_span"),
        _Config(labels_decoder="decoder-model"),
    ],
)
def test_base_gliner_rejects_runtime_unsupported_architectures(runtime, config):
    with pytest.raises(NotImplementedError, match="does not support"):
        BaseGLiNER._validate_runtime_architecture(config, runtime)
