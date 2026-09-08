import pytest
import transformers

pytest.importorskip("accelerate")

from gliner.model import BaseGLiNER
from gliner.training import TrainingArguments


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 11),
        ({"warmup_steps": 7}, 7),
        ({"warmup_ratio": 0.25}, 26),
        ({"warmup_ratio": 0.25, "warmup_steps": 7}, 7),
        ({"warmup_ratio": 0.0}, 0),
        ({"warmup_ratio": 1.0}, 101),
        ({"max_steps": -1, "num_train_epochs": 2, "warmup_ratio": 0.25}, 26),
    ],
)
def test_create_training_args_warmup(tmp_path, kwargs, expected):
    args = BaseGLiNER.create_training_args(output_dir=str(tmp_path), use_cpu=True, **kwargs)

    assert args.get_warmup_steps(101) == expected


def test_direct_training_arguments_warmup(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", warmup_ratio=0.25)

    assert args.get_warmup_steps(101) == 26
    assert args.to_dict()["warmup_ratio"] == 0.25


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_invalid_warmup_ratio(tmp_path, ratio):
    with pytest.raises(ValueError, match="warmup_ratio"):
        BaseGLiNER.create_training_args(output_dir=str(tmp_path), use_cpu=True, warmup_ratio=ratio)


@pytest.mark.skipif(
    "warmup_ratio" in transformers.TrainingArguments.__dataclass_fields__,
    reason="Fractional warmup_steps requires the new Transformers warmup API",
)
def test_fractional_warmup_steps_takes_precedence(tmp_path):
    args = BaseGLiNER.create_training_args(output_dir=str(tmp_path), use_cpu=True, warmup_steps=0.2)

    assert args.get_warmup_steps(101) == 21
