import pickle
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from gliner.model import (
    BaseEncoderGLiNER,
    BaseBiEncoderGLiNER,
    StreamingSpanGLiNER,
    UniEncoderSpanRelexGLiNER,
    UniEncoderTokenDecoderGLiNER,
    _attach_entity_vectors,
    _attach_relation_vectors,
)
from gliner.modeling.base import UniEncoderTokenDecoderModel
from gliner.decoding.decoder import (
    Span,
    SpanDecoder,
    TokenDecoder,
    DecodedRelation,
    SpanRelexDecoder,
    TokenGenerativeDecoder,
)
from gliner.modeling.outputs import GLiNERBaseOutput


def test_span_decoder_preserves_flat_span_and_class_indices():
    decoder = SpanDecoder(SimpleNamespace(max_width=2))
    logits = torch.full((1, 2, 2, 2), -10.0)
    logits[0, 1, 0, 1] = 10.0

    decoded = decoder.decode(
        tokens=[["zero", "one"]],
        id_to_classes={1: "first", 2: "second"},
        model_output=logits,
        threshold=0.5,
    )

    assert len(decoded[0]) == 1
    span = decoded[0][0]
    assert (span.start, span.end, span.entity_type) == (1, 1, "second")
    assert span.class_index == 1
    assert span.span_index == 2  # start * max_width + zero-based width


def test_token_decoder_preserves_class_index_without_inventing_a_span_index():
    decoder = TokenDecoder(SimpleNamespace())
    logits = torch.full((1, 3, 2, 3), -10.0)
    logits[0, 0, 1, 0] = 10.0  # start, class index 1
    logits[0, 0, 1, 2] = 10.0  # inside
    logits[0, 1, 1, 2] = 10.0  # inside
    logits[0, 1, 1, 1] = 10.0  # end

    decoded = decoder.decode(
        tokens=[["New", "York", "today"]],
        id_to_classes={1: "person", 2: "location"},
        model_output=logits,
        threshold=0.5,
    )

    assert len(decoded[0]) == 1
    span = decoded[0][0]
    assert (span.start, span.end, span.entity_type) == (0, 1, "location")
    assert span.class_index == 1
    assert span.span_index is None


def test_span_relex_decoder_preserves_relation_pair_and_class_indices():
    decoder = SpanRelexDecoder(SimpleNamespace(max_width=1))
    entity_logits = torch.full((1, 2, 1, 1), -10.0)
    entity_logits[0, 0, 0, 0] = 10.0
    entity_logits[0, 1, 0, 0] = 10.0
    relation_logits = torch.full((1, 2, 2), -10.0)
    relation_logits[0, 1, 1] = 10.0

    spans, relations = decoder.decode(
        tokens=[["Alice", "Acme"]],
        id_to_classes={1: "entity"},
        model_output=entity_logits,
        rel_idx=torch.tensor([[[0, 1], [1, 0]]]),
        rel_logits=relation_logits,
        rel_mask=torch.tensor([[True, True]]),
        rel_id_to_classes={1: "first relation", 2: "second relation"},
        threshold=0.5,
        relation_threshold=0.5,
    )

    assert len(spans[0]) == 2
    assert len(relations[0]) == 1
    relation = relations[0][0]
    assert isinstance(relation, DecodedRelation)
    assert tuple(relation)[:3] == (1, "second relation", 0)
    assert relation.pair_index == 1
    assert relation.class_index == 1


def test_base_output_omits_span_embeddings_by_default_and_exposes_them_when_requested():
    default_output = GLiNERBaseOutput(logits=torch.zeros(1))

    assert default_output.span_embeddings is None
    assert "span_embeddings" not in default_output

    span_embeddings = torch.ones(1, 2, 3)
    enriched_output = GLiNERBaseOutput(logits=torch.zeros(1), span_embeddings=span_embeddings)

    assert enriched_output.span_embeddings is span_embeddings
    assert enriched_output["span_embeddings"] is span_embeddings


def test_span_vectors_gather_exact_native_rows_and_convert_to_float32():
    span_embeddings = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4)
    decoded = [
        [
            Span(
                start=1,
                end=2,
                entity_type="organization",
                score=0.9,
                class_index=1,
                span_index=4,
            )
        ]
    ]

    _attach_entity_vectors(
        decoded,
        SimpleNamespace(span_embeddings=span_embeddings),
        SimpleNamespace(span_mode="span_level"),
        return_vectors=True,
        return_label_vectors=False,
    )

    vector = decoded[0][0].vector
    np.testing.assert_array_equal(vector, span_embeddings.flatten(1, 2)[0, 4].numpy())
    assert vector.dtype == np.float32
    assert decoded[0][0].label_vector is None


def test_token_vectors_average_inclusive_word_ranges_and_select_each_class_index():
    words_embedding = torch.tensor(
        [[[-3.0, 1.0], [1.0, 5.0], [8.0, 11.0]]],
        dtype=torch.float64,
    )
    prompts_embedding = torch.tensor(
        [[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]],
        dtype=torch.float64,
    )
    first = Span(0, 1, "first", 0.9, class_index=0)
    second = Span(2, 2, "third", 0.8, class_index=2)

    _attach_entity_vectors(
        [[first, second]],
        SimpleNamespace(
            words_embedding=words_embedding,
            prompts_embedding=prompts_embedding,
        ),
        SimpleNamespace(span_mode="token_level"),
        return_vectors=True,
        return_label_vectors=True,
    )

    # Span ends are inclusive: positions 0 and 1 both contribute to the first mean.
    np.testing.assert_array_equal(first.vector, np.array([-1.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(second.vector, np.array([8.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(first.label_vector, np.array([10.0, 11.0], dtype=np.float32))
    np.testing.assert_array_equal(second.label_vector, np.array([30.0, 31.0], dtype=np.float32))
    assert first.vector.dtype == first.label_vector.dtype == np.float32


def test_label_vectors_can_be_requested_without_span_vectors():
    first = Span(0, 0, "public name one", 0.9, class_index=0, span_index=0)
    second = Span(0, 0, "public name two", 0.8, class_index=1, span_index=0)
    prompts_embedding = torch.tensor([[[1.0, 2.0], [7.0, 8.0]]])

    _attach_entity_vectors(
        [[first, second]],
        SimpleNamespace(prompts_embedding=prompts_embedding),
        SimpleNamespace(span_mode="span_level"),
        return_vectors=False,
        return_label_vectors=True,
    )

    assert first.vector is None
    assert second.vector is None
    np.testing.assert_array_equal(first.label_vector, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(second.label_vector, np.array([7.0, 8.0], dtype=np.float32))


def test_entity_mapper_only_includes_vector_keys_when_attached():
    vector = np.array([1.0, 2.0], dtype=np.float32)
    label_vector = np.array([3.0, 4.0], dtype=np.float32)
    plain = Span(0, 0, "person", 0.9)
    enriched = Span(1, 1, "organization", 0.8, vector=vector, label_vector=label_vector)

    mapped = BaseEncoderGLiNER._map_entities_to_original(
        None,
        [[plain, enriched]],
        [0],
        [[0, 6]],
        [[5, 10]],
        ["Alice Acme"],
        1,
    )[0]

    assert "vector" not in mapped[0]
    assert "label_vector" not in mapped[0]
    assert mapped[1]["vector"] is vector
    assert mapped[1]["label_vector"] is label_vector


def test_pair_relation_vectors_and_labels_are_selected_independently():
    first = DecodedRelation(0, "works for", 1, 0.9, pair_index=1, class_index=0)
    second = DecodedRelation(1, "located in", 0, 0.8, pair_index=0, class_index=1)
    relation_embeddings = torch.tensor(
        [[[1.0, 2.0], [5.0, 6.0]]],
        dtype=torch.float64,
    )
    rel_prompts_embedding = torch.tensor(
        [[[11.0, 12.0], [21.0, 22.0]]],
        dtype=torch.float64,
    )

    _attach_relation_vectors(
        [[first, second]],
        SimpleNamespace(
            relation_embeddings=relation_embeddings,
            rel_prompts_embedding=rel_prompts_embedding,
        ),
        return_vectors=True,
        return_label_vectors=True,
    )

    np.testing.assert_array_equal(first.vector, np.array([5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(second.vector, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(first.label_vector, np.array([11.0, 12.0], dtype=np.float32))
    np.testing.assert_array_equal(second.label_vector, np.array([21.0, 22.0], dtype=np.float32))
    assert first.vector.dtype == first.label_vector.dtype == np.float32
    assert first.head_relation_vector is None
    assert first.tail_relation_vector is None


def test_triple_relation_vectors_attach_separate_head_and_tail_rows():
    relation = DecodedRelation(0, "works for", 1, 0.9, pair_index=1, class_index=0)
    head_embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
    tail_embeddings = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float64)

    _attach_relation_vectors(
        [[relation]],
        SimpleNamespace(
            relation_head_embeddings=head_embeddings,
            relation_tail_embeddings=tail_embeddings,
        ),
        return_vectors=True,
        return_label_vectors=False,
    )

    assert relation.vector is None
    assert relation.label_vector is None
    np.testing.assert_array_equal(
        relation.head_relation_vector,
        np.array([3.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        relation.tail_relation_vector,
        np.array([7.0, 8.0], dtype=np.float32),
    )
    assert relation.head_relation_vector.dtype == relation.tail_relation_vector.dtype == np.float32


def test_relation_mapper_preserves_pair_and_triple_vectors_and_omits_missing_keys():
    entities = [[Span(0, 0, "person", 0.9), Span(1, 1, "organization", 0.8)]]
    pair = DecodedRelation(
        0,
        "works for",
        1,
        0.7,
        vector=np.array([1.0], dtype=np.float32),
        label_vector=np.array([2.0], dtype=np.float32),
    )
    triple = DecodedRelation(
        1,
        "employs",
        0,
        0.6,
        head_relation_vector=np.array([3.0], dtype=np.float32),
        tail_relation_vector=np.array([4.0], dtype=np.float32),
    )
    plain = DecodedRelation(0, "knows", 1, 0.5)

    mapped = UniEncoderSpanRelexGLiNER._process_relations(
        None,
        [[pair, triple, plain]],
        entities,
        [[0, 6]],
        [[5, 10]],
        ["Alice Acme"],
    )[0]

    assert mapped[0]["vector"] is pair.vector
    assert mapped[0]["label_vector"] is pair.label_vector
    assert "head_relation_vector" not in mapped[0]
    assert "tail_relation_vector" not in mapped[0]

    assert mapped[1]["head_relation_vector"] is triple.head_relation_vector
    assert mapped[1]["tail_relation_vector"] is triple.tail_relation_vector
    assert "vector" not in mapped[1]
    assert "label_vector" not in mapped[1]

    for key in ("vector", "label_vector", "head_relation_vector", "tail_relation_vector"):
        assert key not in mapped[2]


def test_decoded_relation_keeps_legacy_tuple_payload_and_metadata_when_pickled():
    relation = DecodedRelation(
        2,
        "parent of",
        4,
        0.75,
        pair_index=6,
        class_index=3,
        vector=np.array([1.0, 2.0], dtype=np.float32),
    )

    assert tuple(relation) == (2, "parent of", 4, 0.75)
    assert len(relation) == 4
    assert (relation.head_idx, relation.label, relation.tail_idx, relation.score) == tuple(relation)
    assert relation.pair_index == 6
    assert relation.class_index == 3

    restored = pickle.loads(pickle.dumps(relation))
    assert tuple(restored) == tuple(relation)
    assert restored.pair_index == relation.pair_index
    assert restored.class_index == relation.class_index
    np.testing.assert_array_equal(restored.vector, relation.vector)


def test_span_internal_vector_metadata_does_not_change_legacy_equality_or_repr():
    plain = Span(0, 1, "location", 0.9)
    enriched = Span(
        0,
        1,
        "location",
        0.9,
        class_index=2,
        span_index=4,
        vector=np.array([1.0, 2.0], dtype=np.float32),
        label_vector=np.array([3.0, 4.0], dtype=np.float32),
    )

    assert enriched == plain
    assert repr(enriched) == repr(plain)


def test_encoder_inference_propagates_vector_flags_to_batch_processing():
    model = BaseEncoderGLiNER.__new__(BaseEncoderGLiNER)
    torch.nn.Module.__init__(model)
    model.runtime_model = False
    model._inference_packing_config = None
    prepared = {
        "input_x": [{"row": 0}],
        "entity_types": ["person"],
        "label_names": ["person"],
        "valid_texts": ["Alice"],
        "valid_to_orig_idx": [0],
        "start_token_map": [[0]],
        "end_token_map": [[5]],
        "word_input_spans": None,
        "num_original": 1,
    }
    observed = {}

    model.prepare_batch = lambda *args, **kwargs: prepared
    model.create_collator = object
    model.collate_batch = lambda *args, **kwargs: {"tokens": [["Alice"]], "id_to_classes": {1: "person"}}

    def process_batches(data_loader, *args, **kwargs):
        list(data_loader)
        observed.update(kwargs)
        return [[]]

    model._process_batches = process_batches
    model.map_entities_to_text = lambda decoded, *args: decoded

    assert model.inference(
        ["Alice"],
        ["person"],
        return_vectors=True,
        return_label_vectors=True,
    ) == [[]]
    assert observed["return_vectors"] is True
    assert observed["return_label_vectors"] is True


def test_relex_predict_relations_propagates_vector_flags_to_inference():
    model = UniEncoderSpanRelexGLiNER.__new__(UniEncoderSpanRelexGLiNER)
    torch.nn.Module.__init__(model)
    observed = {}

    def inference(*args, **kwargs):
        observed.update(kwargs)
        return [[{"label": "person"}]], [[{"relation": "works for"}]]

    model.inference = inference

    entities, relations = model.predict_relations(
        "Alice Acme",
        ["person", "organization"],
        ["works for"],
        return_vectors=True,
        return_label_vectors=True,
    )

    assert entities == [{"label": "person"}]
    assert relations == [{"relation": "works for"}]
    assert observed["return_relations"] is True
    assert observed["return_vectors"] is True
    assert observed["return_label_vectors"] is True


def test_bi_encoder_cached_inference_uses_native_embedding_argument():
    model = BaseBiEncoderGLiNER.__new__(BaseBiEncoderGLiNER)
    torch.nn.Module.__init__(model)
    model.runtime_model = False
    observed = {}

    def inference(*args, **kwargs):
        observed.update(kwargs)
        return [[]]

    model.inference = inference
    embeddings = torch.randn(2, 4)

    assert model.batch_predict_with_embeds(
        ["Alice"],
        embeddings,
        ["person", "organization"],
        return_label_vectors=True,
    ) == [[]]
    assert observed["labels_embeds"] is embeddings
    assert "labels_embeddings" not in observed
    assert observed["return_label_vectors"] is True


def test_token_generative_decode_returns_mean_pooled_and_label_vectors():
    config = SimpleNamespace(
        span_mode="token_level",
        labels_decoder="enabled",
        decoder_mode="span",
    )
    model = UniEncoderTokenDecoderGLiNER.__new__(UniEncoderTokenDecoderGLiNER)
    torch.nn.Module.__init__(model)
    model.config = config
    model.decoder = TokenGenerativeDecoder(config)
    output = SimpleNamespace(
        logits=torch.zeros(1, 2, 1, 3),
        gen_labels=["generated label"],
        decoder_span_idx=torch.tensor([[0]]),
        num_gen_sequences=1,
        span_logits=torch.tensor([[[10.0], [-10.0]]]),
        span_idx=torch.tensor([[[0, 1], [1, 1]]]),
        span_mask=torch.tensor([[True, True]]),
        words_embedding=torch.tensor([[[1.0, 3.0], [5.0, 7.0]]]),
        prompts_embedding=torch.tensor([[[11.0, 13.0]]]),
        span_embeddings=None,
    )

    decoded = model.decode_batch(
        output,
        {"tokens": [["New", "York"]], "id_to_classes": {1: "location"}},
        return_vectors=True,
        return_label_vectors=True,
    )

    assert len(decoded[0]) == 1
    span = decoded[0][0]
    assert (span.start, span.end, span.generated_labels) == (0, 1, ["generated label"])
    np.testing.assert_array_equal(span.vector, np.array([3.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(span.label_vector, np.array([11.0, 13.0], dtype=np.float32))


def test_token_generative_span_selection_uses_the_boolean_span_mask():
    model = UniEncoderTokenDecoderModel.__new__(UniEncoderTokenDecoderModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(decoder_mode="span")

    selected, selected_mask, selected_indices = model.select_token_decoder_embedding(
        prompts_embedding=torch.zeros(1, 1, 2),
        prompts_embedding_mask=torch.ones(1, 1),
        span_logits=torch.tensor([[[10.0], [-10.0]]]),
        span_rep=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        span_idx=torch.tensor([[[0, 0], [1, 1]]]),
        span_mask=torch.tensor([[True, True]]),
    )

    assert selected.shape == (1, 2, 1, 2)
    assert selected_mask.squeeze(-1).tolist() == [[1, 0]]
    assert selected_indices.tolist() == [[0, -1]]


@pytest.mark.parametrize("flag", ["return_vectors", "return_label_vectors"])
def test_runtime_inference_rejects_unexported_vector_outputs(flag):
    model = BaseEncoderGLiNER.__new__(BaseEncoderGLiNER)
    torch.nn.Module.__init__(model)
    model.runtime_model = True

    with pytest.raises(NotImplementedError, match="ONNX/OpenVINO"):
        model.inference(["Alice"], ["person"], **{flag: True})


def test_stateful_streaming_rejects_vector_outputs_before_mutating_a_session():
    model = StreamingSpanGLiNER.__new__(StreamingSpanGLiNER)
    torch.nn.Module.__init__(model)

    with pytest.raises(NotImplementedError, match="stateful streaming"):
        model.inference(
            ["Alice"],
            ["person"],
            session_id=["session-1"],
            return_vectors=True,
        )
