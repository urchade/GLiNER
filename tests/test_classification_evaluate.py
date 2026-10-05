import sys
import importlib
from types import ModuleType
from unittest.mock import patch

import pytest

try:
    datasets_module = importlib.import_module("datasets")
except ModuleNotFoundError as error:
    if error.name != "datasets":
        raise

    datasets_module = ModuleType("datasets")

    class Dataset:
        def __init__(self, columns):
            self.columns = columns

        @classmethod
        def from_dict(cls, columns):
            return cls(columns)

    def _unavailable_load_dataset(*args, **kwargs):
        raise AssertionError("load_dataset should be mocked by each test")

    datasets_module.Dataset = Dataset
    datasets_module.load_dataset = _unavailable_load_dataset
    sys.modules["datasets"] = datasets_module

try:
    importlib.import_module("sklearn.metrics")
except ModuleNotFoundError as error:
    if error.name not in {"sklearn", "sklearn.metrics"}:
        raise

    sklearn_module = ModuleType("sklearn")
    metrics_module = ModuleType("sklearn.metrics")

    def _unused_f1_score(*args, **kwargs):
        raise AssertionError("the classifier stub does not calculate F1 scores")

    metrics_module.f1_score = _unused_f1_score
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module

Dataset = datasets_module.Dataset
GLiNERClassifier = importlib.import_module("gliner.multitask.classification").GLiNERClassifier


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
