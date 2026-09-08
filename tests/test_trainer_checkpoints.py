from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest
import transformers
from torch import nn
from safetensors.torch import save_file

pytest.importorskip("accelerate")

from gliner.model import BaseGLiNER
from gliner.config import GLiNERConfig
from gliner.training import Trainer, TrainingArguments


class TinyGLiNER(BaseGLiNER):
    _create_model = None
    _create_data_processor = None
    resize_embeddings = None
    inference = None
    evaluate = None

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.ModuleDict({"encoder": nn.Linear(2, 2), "head": nn.Linear(2, 2)})
        self.model["head"].weight = self.model["encoder"].weight
        self.model.register_buffer("count", torch.tensor(7))
        self.config = GLiNERConfig()
        self.data_processor = SimpleNamespace(transformer_tokenizer=Mock())
        self._keys_to_ignore_on_save = None


def make_trainer(tmp_path, model=None):
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    return Trainer(model=TinyGLiNER() if model is None else model, args=args)


@pytest.mark.parametrize("safe_serialization", [False, True])
@pytest.mark.parametrize("load_method", ["best", "resume", "explicit_model"])
def test_checkpoint_restores_weights(tmp_path, caplog, safe_serialization, load_method):
    trainer = make_trainer(tmp_path)
    trainer.args.save_safetensors = safe_serialization
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    checkpoint = tmp_path / "checkpoint-1"
    trainer.save_model(str(checkpoint))
    assert (checkpoint / "gliner_config.json").is_file()
    assert not (checkpoint / "config.json").exists()

    target = TinyGLiNER() if load_method == "explicit_model" else trainer.model
    parameters = list(target.parameters())
    with torch.no_grad():
        for tensor in target.state_dict().values():
            tensor.zero_()

    if load_method == "best":
        trainer.state.best_model_checkpoint = str(checkpoint)
        trainer._load_best_model()
    elif load_method == "explicit_model":
        trainer._load_from_checkpoint(str(checkpoint), model=target)
    else:
        trainer._load_from_checkpoint(str(checkpoint))

    for key, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, expected[key])
    assert all(before is after for before, after in zip(parameters, target.parameters(), strict=True))
    assert target.model["head"].weight is target.model["encoder"].weight
    assert "missing keys" not in caplog.text
    assert "unexpected keys" not in caplog.text


@pytest.mark.parametrize("load_method", ["best", "resume"])
@pytest.mark.parametrize("gliner_model", [False, True])
def test_standard_checkpoint_restores_wrapper_weights(tmp_path, caplog, load_method, gliner_model):
    model = TinyGLiNER() if gliner_model else nn.Linear(2, 2)
    trainer = make_trainer(tmp_path, model)
    expected = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    save_file(expected, tmp_path / "model.safetensors")
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.zero_()

    if load_method == "best":
        trainer.state.best_model_checkpoint = str(tmp_path)
        trainer._load_best_model()
    else:
        trainer._load_from_checkpoint(str(tmp_path))

    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[key])
    assert "missing keys" not in caplog.text
    assert "unexpected keys" not in caplog.text


def test_incomplete_checkpoint_still_reports_incompatible_keys(tmp_path, caplog):
    trainer = make_trainer(tmp_path)
    trainer.save_model(str(tmp_path))
    state_dict = {key: tensor.clone() for key, tensor in trainer.model.model.state_dict().items()}
    del state_dict["encoder.bias"]
    state_dict["unexpected"] = torch.zeros(1)
    save_file(state_dict, tmp_path / "model.safetensors")

    trainer._load_from_checkpoint(str(tmp_path))

    assert "missing keys" in caplog.text
    assert "encoder.bias" in caplog.text
    assert "unexpected keys" in caplog.text


@pytest.mark.parametrize("load_method", ["_load_best_model", "_load_from_checkpoint"])
@pytest.mark.parametrize("backend", ["is_deepspeed_enabled", "is_fsdp_enabled", "sagemaker"])
def test_distributed_checkpoint_loading_uses_transformers(tmp_path, monkeypatch, load_method, backend):
    trainer = make_trainer(tmp_path)
    trainer.save_model(str(tmp_path))
    if backend == "sagemaker":
        monkeypatch.setattr("transformers.trainer.is_sagemaker_mp_enabled", lambda: True)
    else:
        setattr(trainer, backend, True)
    parent_load = Mock()
    monkeypatch.setattr(transformers.Trainer, load_method, parent_load)

    if load_method == "_load_best_model":
        trainer.state.best_model_checkpoint = str(tmp_path)
        trainer._load_best_model()
        parent_load.assert_called_once_with()
    else:
        trainer._load_from_checkpoint(str(tmp_path), model=trainer.model)
        parent_load.assert_called_once_with(str(tmp_path), model=trainer.model)
