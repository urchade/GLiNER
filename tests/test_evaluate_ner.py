import json

import torch

from gliner.evaluation.evaluate_ner import create_dataset, get_for_all_path


class _ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def evaluate(self, test_data, **kwargs):
        return "P: 100.00%\tR: 100.00%\tF1: 100.00%\n", 1.0


def _write_dataset(path):
    path.mkdir(parents=True)
    sample = [{"sentence": "Paris is nice", "entities": [{"pos": [0, 5], "type": "LOC"}]}]
    for split in ("train", "dev", "test"):
        (path / f"{split}.json").write_text(json.dumps(sample))
    (path / "labels.json").write_text(json.dumps(["LOC"]))


def test_create_dataset_matches_split_names_on_file_name(tmp_path):
    dataset = tmp_path / "dev-benchmarks" / "conll"
    _write_dataset(dataset)

    train, dev, test, labels = create_dataset(str(dataset))

    assert len(train) == len(dev) == len(test) == 1
    assert test[0]["ner"] == [(0, 0, "loc")]
    assert labels == ["loc"]


def test_get_for_all_path_without_zero_shot_datasets(tmp_path):
    _write_dataset(tmp_path / "data" / "conll")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    get_for_all_path(_ConstantModel(), 0, str(log_dir), str(tmp_path / "data"))

    tables = (log_dir / "tables.txt").read_text()
    assert "conll               : 100.0%" in tables
    assert "Table for zero-shot benchmark" not in tables
