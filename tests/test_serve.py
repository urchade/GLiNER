from unittest.mock import Mock

import pytest

from gliner.serve.client import GLiNERClient
from gliner.serve.server import GLiNERServer, _min_batch_value, _normalize_relation_lists


class _FakeModel:
    def __init__(self):
        self.run_batch_kwargs = None
        self.decode_batch_kwargs = None
        self.prepared_labels = None
        self.collated_label_names = None

    def prepare_batch(self, texts, labels):
        self.prepared_labels = labels
        valid_texts = [text for text in texts if text.strip()]
        valid_to_orig_idx = [idx for idx, text in enumerate(texts) if text.strip()]
        return {
            "input_x": [{"tokenized_text": text.split(), "ner": None} for text in valid_texts],
            "entity_types": labels,
            "label_names": labels,
            "valid_texts": valid_texts,
            "valid_to_orig_idx": valid_to_orig_idx,
            "start_token_map": [[0] for _ in valid_texts],
            "end_token_map": [[1] for _ in valid_texts],
            "num_original": len(texts),
        }

    def collate_batch(self, input_x, entity_types, collator, label_names=None):
        self.collated_label_names = label_names
        return {"input_x": input_x, "entity_types": entity_types}

    def run_batch(self, batch, **kwargs):
        self.run_batch_kwargs = kwargs
        return object()

    def decode_batch(self, model_output, batch, **kwargs):
        self.decode_batch_kwargs = kwargs
        return [[object()] for _ in batch["input_x"]]

    def map_entities_to_text(
        self, decoded, valid_texts, valid_to_orig_idx, start_token_map, end_token_map, num_original
    ):
        results = [[] for _ in range(num_original)]
        for decoded_idx, original_idx in enumerate(valid_to_orig_idx):
            results[original_idx] = decoded[decoded_idx]
        return results


def test_min_batch_value_uses_lowest_threshold_for_model_pruning():
    assert _min_batch_value([0.8, 0.2, 0.5]) == 0.2
    assert _min_batch_value(0.7) == 0.7


def test_normalize_relation_lists_handles_heterogeneous_optional_relations():
    assert _normalize_relation_lists([None, ["works_at"], []]) == [[], ["works_at"], []]
    assert _normalize_relation_lists([None, []]) is None
    assert _normalize_relation_lists(["works_at"]) == ["works_at"]
    assert _normalize_relation_lists(None) is None


def test_observed_seq_len_uses_largest_per_request_prompt():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(calibration_min_seq_len=1, max_model_len=512)

    observed = server.observed_seq_len(
        ["short text", "this is the longest text"],
        labels=[["person"], ["large language model", "organization"]],
        relations=[None, ["founded by"]],
    )

    assert observed == 11


def test_observed_seq_len_uses_dictionary_descriptions():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(calibration_min_seq_len=1, max_model_len=512)

    observed = server.observed_seq_len(
        ["short text", "longer text here"],
        labels=[
            {"person": "a human individual"},
            {"org": "a company or other organization", "place": "a geographical place"},
        ],
    )

    assert observed == 11


def test_filter_labels_truncates_per_request_label_lists():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(max_labels=2)

    assert server._filter_labels(
        [
            ["person", "organization", "location"],
            ["date"],
        ]
    ) == [
        ["person", "organization"],
        ["date"],
    ]


def test_filter_labels_truncates_dictionary_labels_without_losing_descriptions():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(max_labels=1)

    assert server._filter_labels(
        [
            {"person": "a human individual", "organization": "a company"},
            {"location": "a geographical place", "date": "a calendar date"},
        ]
    ) == [
        {"person": "a human individual"},
        {"location": "a geographical place"},
    ]


def test_run_batch_ner_passes_heterogeneous_decode_controls():
    server = GLiNERServer.__new__(GLiNERServer)
    server.model = _FakeModel()
    server.collator = object()
    server.packing_config = None
    server.config = Mock(enable_polylora=False, polylora_base_adapter_id="__base__")

    result = server._run_batch_ner(
        ["John works at Acme", "Paris is sunny"],
        [["person", "organization"], ["location"]],
        threshold=[0.2, 0.8],
        flat_ner=[True, False],
        multi_label=[False, True],
    )

    assert len(result) == 2
    assert server.model.prepared_labels == [["person", "organization"], ["location"]]
    assert server.model.collated_label_names == [["person", "organization"], ["location"]]
    assert server.model.run_batch_kwargs["threshold"] == 0.2
    assert server.model.decode_batch_kwargs["threshold"] == [0.2, 0.8]
    assert server.model.decode_batch_kwargs["flat_ner"] == [True, False]
    assert server.model.decode_batch_kwargs["multi_label"] == [False, True]


def test_resolve_adapter_ids_uses_base_adapter_for_empty_request_ids():
    server = GLiNERServer.__new__(GLiNERServer)
    server.config = Mock(enable_polylora=True, polylora_base_adapter_id="__base__")

    assert server._resolve_adapter_ids([None, "task-a", None], [0, 2]) == ["__base__", "__base__"]


def test_run_batch_ner_passes_valid_text_adapter_ids():
    server = GLiNERServer.__new__(GLiNERServer)
    server.model = _FakeModel()
    server.collator = object()
    server.packing_config = None
    server.config = Mock(enable_polylora=True, polylora_base_adapter_id="__base__")

    result = server._run_batch_ner(
        ["John works at Acme", ""],
        ["person", "organization"],
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
        adapter_ids=[None, "task-b"],
    )

    assert len(result) == 2
    assert server.model.run_batch_kwargs["adapter_ids"] == ["__base__"]


def test_client_sends_one_dictionary_label_set_per_text():
    client = GLiNERClient()
    client._post = lambda payload: payload

    results = client.predict(
        ["John works at Acme", "Paris is sunny"],
        labels=[
            {"person": "a human individual", "organization": "a company"},
            {"location": "a geographical place"},
        ],
    )

    assert [result["labels"] for result in results] == [
        {"person": "a human individual", "organization": "a company"},
        {"location": "a geographical place"},
    ]


def test_client_reuses_shared_dictionary_labels_for_each_text():
    client = GLiNERClient()
    client._post = lambda payload: payload
    labels = {"person": "a human individual"}

    results = client.predict(["John", "Jane"], labels=labels)

    assert [result["labels"] for result in results] == [labels, labels]


def test_client_rejects_wrong_number_of_per_text_label_sets():
    client = GLiNERClient()

    with pytest.raises(ValueError, match="Per-text labels must have length 2, got 1"):
        client.predict(["John", "Paris"], labels=[{"person": "a human individual"}])


def test_runtime_server_splits_per_text_labels_into_singleton_batches():
    server = GLiNERServer.__new__(GLiNERServer)
    server.model = Mock(is_runtime_model=True)
    server._supports_relations = False
    calls = []

    def run_batch(texts, labels, threshold, flat_ner, multi_label, adapter_ids):
        calls.append((texts, labels, threshold))
        return [[{"label": next(iter(labels))}]]

    server._run_batch_ner = run_batch
    results = server._run_batch_internal(
        ["Alice", "Paris"],
        [
            {"person": "a human individual"},
            {"location": "a geographical place"},
        ],
        threshold=[0.4, 0.6],
    )

    assert calls == [
        (["Alice"], {"person": "a human individual"}, 0.4),
        (["Paris"], {"location": "a geographical place"}, 0.6),
    ]
    assert results == [[{"label": "person"}], [{"label": "location"}]]
