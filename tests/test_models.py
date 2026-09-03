import torch
import pytest

from gliner import GLiNER
from gliner.model import (
    BaseEncoderGLiNER,
    BiEncoderSpanGLiNER,
    UniEncoderSpanRelexGLiNER,
    _entity_types_for_chunk,
)
from gliner.data_processing import BiEncoderSpanProcessor


class _WordsSplitter:
    def __call__(self, text):
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            cursor = end
            yield token, start, end


def _minimal_encoder_model(cls=BaseEncoderGLiNER):
    model = cls.__new__(cls)
    model.data_processor = type("Processor", (), {"words_splitter": _WordsSplitter()})()
    return model


def test_span_model():
    model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")

    text = """
    Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
    """

    labels = ["person", "award", "date", "competitions", "teams", "person"]

    entities = model.predict_entities(text, labels)

    assert len(entities) > 0


def test_prepare_batch_aligns_per_text_labels_after_empty_text_filtering():
    model = _minimal_encoder_model()

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["valid_to_orig_idx"] == [1]
    assert prepared["entity_types"] == [["person", "organization"]]


def test_prepare_batch_validates_per_text_label_count():
    model = _minimal_encoder_model()

    with pytest.raises(ValueError, match="Per-text labels must have length 2"):
        model.prepare_batch(["one", "two"], [["label"]])


def test_relex_prepare_batch_aligns_per_text_relations_after_empty_text_filtering():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    prepared = model.prepare_batch(
        ["", "John works at Acme"],
        [["wrong"], ["person", "organization"]],
        relations=[["wrong_relation"], ["works_at"]],
    )

    assert prepared["valid_texts"] == ["John works at Acme"]
    assert prepared["entity_types"] == [["person", "organization"]]
    assert prepared["relation_types"] == [["works_at"]]


def test_relex_prepare_batch_validates_per_text_relation_count():
    model = _minimal_encoder_model(UniEncoderSpanRelexGLiNER)

    with pytest.raises(ValueError, match="Per-text relations must have length 2"):
        model.prepare_batch(
            ["one", "two"],
            [["person"], ["organization"]],
            relations=[["works_at"]],
        )


def test_entity_types_for_chunk_slices_per_row_label_sets():
    """Per-row label sets must follow their rows into each DataLoader chunk.

    The loader hands collate_fn one chunk at a time; without this the decoder indexes the
    whole-batch list with a chunk-local index and rows past the first chunk are decoded against
    another row's labels.
    """
    per_row = [["a"], ["b"], ["c"], ["d"]]
    assert _entity_types_for_chunk(per_row, [2, 3]) == [["c"], ["d"]]
    assert _entity_types_for_chunk(per_row, [0, 1]) == [["a"], ["b"]]


def test_entity_types_for_chunk_passes_shared_label_list_through():
    """A single flat list is shared by every row and must NOT be sliced."""
    shared = ["person", "organization", "location"]
    assert _entity_types_for_chunk(shared, [1, 2]) == shared
    assert _entity_types_for_chunk([], [0]) == []


@pytest.mark.parametrize(
    ("onnx_model", "entity_types", "expected_batch_sizes"),
    [
        (False, [["a"], ["b"], ["c"]], [2, 1]),
        (True, [["a"], ["b"], ["c"]], [1, 1, 1]),
        (True, [["a"], ["a"], ["a"]], [2, 1]),
        (True, ["a", "b"], [2, 1]),
    ],
)
def test_biencoder_onnx_uses_singleton_batches_only_for_per_row_labels(
    onnx_model, entity_types, expected_batch_sizes
):
    """Legacy ONNX graphs need singleton chunks because they cannot consume gather metadata."""
    model = BiEncoderSpanGLiNER.__new__(BiEncoderSpanGLiNER)
    torch.nn.Module.__init__(model)
    model.onnx_model = onnx_model
    model.data_processor = BiEncoderSpanProcessor.__new__(BiEncoderSpanProcessor)
    model._inference_packing_config = None
    texts = ["zero", "one", "two"]
    prepared = {
        "input_x": [{"row": i} for i in range(3)],
        "entity_types": entity_types,
        "valid_texts": texts,
        "valid_to_orig_idx": list(range(3)),
        "start_token_map": [[] for _ in texts],
        "end_token_map": [[] for _ in texts],
        "word_input_spans": None,
        "num_original": 3,
    }
    observed_batch_sizes = []

    model.prepare_batch = lambda *args, **kwargs: prepared
    model.create_collator = object
    model.collate_batch = lambda input_x, labels, collator: {"tokens": input_x}

    def process_batches(data_loader, *args, **kwargs):
        batches = list(data_loader)
        observed_batch_sizes.extend(len(batch["tokens"]) for batch in batches)
        return [[] for _ in range(sum(observed_batch_sizes))]

    def map_entities(decoded, *args):
        return decoded

    model._process_batches = process_batches
    model.map_entities_to_text = map_entities

    model.inference(texts, entity_types, batch_size=2)

    assert observed_batch_sizes == expected_batch_sizes


def test_onnx_run_batch_rejects_unaligned_per_row_label_layout():
    """Low-level ONNX batching must fail loudly instead of ignoring gather metadata."""
    model = BiEncoderSpanGLiNER.__new__(BiEncoderSpanGLiNER)
    torch.nn.Module.__init__(model)
    model.onnx_model = True
    batch = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "labels_gather_indices": torch.tensor([[0], [1]]),
    }

    with pytest.raises(ValueError, match="Batched per-row labels are not supported"):
        model.run_batch(batch)


def test_relex_inference_slices_per_row_relation_types_by_chunk():
    """Every relex DataLoader chunk should receive the relation schemas for its own rows."""
    model = UniEncoderSpanRelexGLiNER.__new__(UniEncoderSpanRelexGLiNER)
    torch.nn.Module.__init__(model)
    model._inference_packing_config = None

    texts = ["zero", "one", "two", "three"]
    prepared = {
        "input_x": [{"row": i} for i in range(4)],
        "entity_types": [[f"entity_{i}"] for i in range(4)],
        "relation_types": [[f"relation_{i}"] for i in range(4)],
        "valid_texts": texts,
        "valid_to_orig_idx": list(range(4)),
        "start_token_map": [[] for _ in texts],
        "end_token_map": [[] for _ in texts],
        "word_input_spans": None,
        "num_original": 4,
    }
    received_types = []

    def prepare_batch(*args, **kwargs):
        return prepared

    def create_collator():
        return object()

    def collate_batch(input_x, entity_types, collator, relation_types):
        received_types.append((entity_types, relation_types))
        return {"tokens": [[str(item["row"])] for item in input_x]}

    def process_batches(data_loader, *args, **kwargs):
        batches = list(data_loader)
        batch_size = sum(len(batch["tokens"]) for batch in batches)
        return ([[] for _ in range(batch_size)], [[] for _ in range(batch_size)])

    def passthrough(decoded, *args):
        return decoded

    model.prepare_batch = prepare_batch
    model.create_collator = create_collator
    model.collate_batch = collate_batch
    model._process_batches = process_batches
    model.map_entities_to_text = passthrough
    model.map_relations_to_text = passthrough

    model.inference(
        texts,
        prepared["entity_types"],
        relations=prepared["relation_types"],
        batch_size=2,
    )

    assert received_types == [
        ([['entity_0'], ['entity_1']], [['relation_0'], ['relation_1']]),
        ([['entity_2'], ['entity_3']], [['relation_2'], ['relation_3']]),
    ]
