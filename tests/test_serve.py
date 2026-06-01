from unittest.mock import Mock

from gliner.serve.server import GLiNERServer, _min_batch_value, _normalize_relation_lists


class _FakeModel:
    def __init__(self):
        self.run_batch_kwargs = None
        self.decode_batch_kwargs = None
        self.prepared_labels = None

    def prepare_batch(self, texts, labels):
        self.prepared_labels = labels
        return {
            "input_x": [{"tokenized_text": text.split(), "ner": None} for text in texts],
            "entity_types": labels,
            "valid_texts": texts,
            "valid_to_orig_idx": list(range(len(texts))),
            "start_token_map": [[0] for _ in texts],
            "end_token_map": [[1] for _ in texts],
            "num_original": len(texts),
        }

    def collate_batch(self, input_x, entity_types, collator):
        return {"input_x": input_x, "entity_types": entity_types}

    def run_batch(self, batch, **kwargs):
        self.run_batch_kwargs = kwargs
        return object()

    def decode_batch(self, model_output, batch, **kwargs):
        self.decode_batch_kwargs = kwargs
        return [[object()] for _ in batch["input_x"]]

    def map_entities_to_text(self, decoded, valid_texts, valid_to_orig_idx, start_token_map, end_token_map, num_original):
        return decoded


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


def test_run_batch_ner_passes_heterogeneous_decode_controls():
    server = GLiNERServer.__new__(GLiNERServer)
    server.model = _FakeModel()
    server.collator = object()
    server.packing_config = None

    result = server._run_batch_ner(
        ["John works at Acme", "Paris is sunny"],
        [["person", "organization"], ["location"]],
        threshold=[0.2, 0.8],
        flat_ner=[True, False],
        multi_label=[False, True],
    )

    assert len(result) == 2
    assert server.model.prepared_labels == [["person", "organization"], ["location"]]
    assert server.model.run_batch_kwargs["threshold"] == 0.2
    assert server.model.decode_batch_kwargs["threshold"] == [0.2, 0.8]
    assert server.model.decode_batch_kwargs["flat_ner"] == [True, False]
    assert server.model.decode_batch_kwargs["multi_label"] == [False, True]
