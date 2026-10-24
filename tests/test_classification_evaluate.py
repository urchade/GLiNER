import sys
import importlib
from types import ModuleType
from unittest.mock import patch

import pytest

import gliner  # noqa: F401  (import torch and transformers before any stub module exists)


def _missing(name):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in {name, name.split(".")[0]}:
            raise
        return True
    return False


class Dataset:
    def __init__(self, columns):
        self.columns = columns

    @classmethod
    def from_dict(cls, columns):
        return cls(columns)


def _unavailable_load_dataset(*args, **kwargs):
    raise AssertionError("load_dataset should be mocked by each test")


def _unused_f1_score(*args, **kwargs):
    raise AssertionError("the classifier stub does not calculate F1 scores")


# gliner.multitask needs optional packages (datasets, scikit-learn, evaluate) the tests do not
# use. Stand-ins for the missing ones exist only while it is imported: left in sys.modules they
# would make other code believe those packages are installed.
_stubs = {}
if _missing("datasets"):
    _stubs["datasets"] = ModuleType("datasets")
    _stubs["datasets"].Dataset = Dataset
    _stubs["datasets"].load_dataset = _unavailable_load_dataset
if _missing("sklearn.metrics"):
    _stubs["sklearn"] = ModuleType("sklearn")
    _stubs["sklearn.metrics"] = ModuleType("sklearn.metrics")
    _stubs["sklearn.metrics"].f1_score = _unused_f1_score
    _stubs["sklearn"].metrics = _stubs["sklearn.metrics"]
if _missing("evaluate"):
    _stubs["evaluate"] = ModuleType("evaluate")

sys.modules.update(_stubs)
try:
    GLiNERClassifier = importlib.import_module("gliner.multitask.classification").GLiNERClassifier
finally:
    for _name in _stubs:
        sys.modules.pop(_name, None)

Dataset = sys.modules["gliner.multitask.classification"].Dataset


class _ClassifierStub:
    def __init__(self):
        self.prepared_dataset = None

    def prepare_dataset(self, dataset, labels=None, max_examples=-1):
        self.prepared_dataset = dataset
        return ["example text"], ["example"], ["example"]

    def __call__(self, texts, classes, threshold):
        return [[{"label": classes[0]}]]

    def compute_f_score(self, predicted_labels, true_labels):
        return {"micro": 1.0}


@pytest.fixture
def local_dataset():
    return Dataset.from_dict({"text": ["example text"], "label": ["example"]})


def test_evaluate_uses_preloaded_dataset_without_dataset_id(local_dataset):
    classifier = _ClassifierStub()

    with patch("gliner.multitask.classification.load_dataset") as load_dataset:
        result = GLiNERClassifier.evaluate(classifier, dataset=local_dataset)

    load_dataset.assert_not_called()
    assert classifier.prepared_dataset is local_dataset
    assert result == {"micro": 1.0}


def test_evaluate_loads_dataset_id_when_dataset_is_absent(local_dataset):
    classifier = _ClassifierStub()

    with patch("gliner.multitask.classification.load_dataset", return_value=local_dataset) as load_dataset:
        result = GLiNERClassifier.evaluate(classifier, dataset_id="local/dataset")

    load_dataset.assert_called_once_with("local/dataset")
    assert classifier.prepared_dataset is local_dataset
    assert result == {"micro": 1.0}


def test_evaluate_prefers_preloaded_dataset_when_both_inputs_are_supplied(local_dataset):
    classifier = _ClassifierStub()

    with patch("gliner.multitask.classification.load_dataset") as load_dataset:
        result = GLiNERClassifier.evaluate(
            classifier,
            dataset_id="ignored/dataset",
            dataset=local_dataset,
        )

    load_dataset.assert_not_called()
    assert classifier.prepared_dataset is local_dataset
    assert result == {"micro": 1.0}


def test_evaluate_requires_either_dataset_or_dataset_id():
    classifier = _ClassifierStub()

    with (
        patch("gliner.multitask.classification.load_dataset") as load_dataset,
        pytest.raises(ValueError, match="Either 'dataset_id' or 'dataset'"),
    ):
        GLiNERClassifier.evaluate(classifier)

    load_dataset.assert_not_called()
    assert classifier.prepared_dataset is None
