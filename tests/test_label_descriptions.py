from types import SimpleNamespace

import torch
import pytest

from gliner.model import BaseEncoderGLiNER, UniEncoderSpanRelexGLiNER
from gliner.decoding.decoder import Span


class _WordsSplitter:
    def __call__(self, text):
        cursor = 0
        for token in text.split():
            start = text.index(token, cursor)
            end = start + len(token)
            cursor = end
            yield token, start, end


class _RecordingCollator:
    """Small stand-in that exposes exactly which strings reach the processor."""

    def __init__(self):
        self.entity_types = []

    def __call__(self, input_x, entity_types, relation_types=None):
        self.entity_types.append(entity_types)

        if entity_types and isinstance(entity_types[0], list):
            id_to_classes = [dict(enumerate(label_set, start=1)) for label_set in entity_types]
        else:
            id_to_classes = dict(enumerate(entity_types, start=1))

        batch = {
            "tokens": [item["tokenized_text"] for item in input_x],
            "id_to_classes": id_to_classes,
        }
        if relation_types is not None:
            batch["rel_id_to_classes"] = dict(enumerate(relation_types, start=1))
        return batch


def _minimal_encoder_model():
    model = BaseEncoderGLiNER.__new__(BaseEncoderGLiNER)
    torch.nn.Module.__init__(model)
    model.data_processor = SimpleNamespace(words_splitter=_WordsSplitter())
    model._inference_packing_config = None
    return model


def _inference_model():
    model = _minimal_encoder_model()
    collator = _RecordingCollator()
    model.create_collator = lambda: collator
    model.run_batch = lambda *args, **kwargs: object()

    def decode_batch(model_output, batch, **kwargs):
        mappings = batch["id_to_classes"]
        if not isinstance(mappings, list):
            mappings = [mappings for _ in batch["tokens"]]

        return [
            [
                Span(
                    start=0,
                    end=0,
                    entity_type=mapping[1],
                    score=0.9,
                    class_probs={mapping[1]: 0.9},
                )
            ]
            for mapping in mappings
        ]

    model.decode_batch = decode_batch
    return model, collator


def test_prepare_batch_splits_dictionary_prompts_from_public_names():
    model = _minimal_encoder_model()

    prepared = model.prepare_batch(
        ["John works at Acme"],
        {
            "person": "A human individual",
            "organization": "A company or institution",
        },
    )

    assert prepared["entity_types"] == ["A human individual", "A company or institution"]
    assert prepared["label_names"] == ["person", "organization"]


def test_prepare_batch_aligns_per_text_dictionaries_after_empty_text_filtering():
    model = _minimal_encoder_model()

    prepared = model.prepare_batch(
        ["", "Paris is sunny", "Acme hired Alice"],
        [
            {"wrong": "Must be removed with the empty text"},
            {"location": "A geographical place"},
            {
                "organization": "A company or institution",
                "person": "A human individual",
            },
        ],
    )

    assert prepared["valid_to_orig_idx"] == [1, 2]
    assert prepared["entity_types"] == [
        ["A geographical place"],
        ["A company or institution", "A human individual"],
    ]
    assert prepared["label_names"] == [
        ["location"],
        ["organization", "person"],
    ]


def test_prepare_batch_validates_per_text_dictionary_count():
    model = _minimal_encoder_model()

    with pytest.raises(ValueError, match="Per-text labels must have length 2"):
        model.prepare_batch(["one", "two"], [{"person": "A human individual"}])


@pytest.mark.parametrize(
    "labels",
    [
        {1: "A human individual"},
        {"person": 1},
        [{"person": "A human individual"}, {"location": 2}],
    ],
)
def test_prepare_batch_rejects_non_string_dictionary_entries(labels):
    model = _minimal_encoder_model()

    with pytest.raises(TypeError, match="string keys and values"):
        model.prepare_batch(["one", "two"], labels)


def test_prepare_batch_rejects_duplicate_descriptions():
    model = _minimal_encoder_model()

    with pytest.raises(ValueError, match="descriptions must be unique"):
        model.prepare_batch(
            ["Alice"],
            {
                "person": "A named thing",
                "organization": "A named thing",
            },
        )


def test_prepare_batch_rejects_descriptions_with_precomputed_prompts():
    model = _minimal_encoder_model()
    model.config = SimpleNamespace(precomputed_prompts_mode=True)

    with pytest.raises(ValueError, match="precomputed prompt embeddings"):
        model.prepare_batch(["Alice"], {"person": "A human individual"})


@pytest.mark.parametrize(
    ("labels", "expected_prompts", "expected_names"),
    [
        (
            {"person": "A human individual", "organization": "A company or institution"},
            ["A human individual", "A company or institution"],
            {1: "person", 2: "organization"},
        ),
        (
            [
                {"person": "A human individual"},
                {"location": "A geographical place"},
            ],
            [["A human individual"], ["A geographical place"]],
            [{1: "person"}, {1: "location"}],
        ),
    ],
)
def test_collate_batch_encodes_descriptions_and_decodes_public_names(labels, expected_prompts, expected_names):
    model = _minimal_encoder_model()
    prepared = model.prepare_batch(["Alice", "Paris"], labels)
    collator = _RecordingCollator()

    batch = model.collate_batch(
        prepared["input_x"],
        prepared["entity_types"],
        collator,
        label_names=prepared["label_names"],
    )

    assert collator.entity_types == [expected_prompts]
    assert batch["id_to_classes"] == expected_names


def test_relex_collate_encodes_descriptions_and_keeps_public_entity_names():
    model = UniEncoderSpanRelexGLiNER.__new__(UniEncoderSpanRelexGLiNER)
    torch.nn.Module.__init__(model)
    model.data_processor = SimpleNamespace(words_splitter=_WordsSplitter())
    prepared = model.prepare_batch(
        ["Alice works at Acme"],
        {"person": "A human individual", "organization": "A company"},
        relations=["works for"],
    )
    collator = _RecordingCollator()

    batch = model.collate_batch(
        prepared["input_x"],
        prepared["entity_types"],
        collator,
        relation_types=prepared["relation_types"],
        label_names=prepared["label_names"],
    )

    assert collator.entity_types == [["A human individual", "A company"]]
    assert batch["id_to_classes"] == {1: "person", 2: "organization"}
    assert batch["rel_id_to_classes"] == {1: "works for"}


@pytest.mark.parametrize(
    ("labels", "expected_prompts", "expected_output_names"),
    [
        (
            {"person": "A human individual", "organization": "A company or institution"},
            [
                ["A human individual", "A company or institution"],
                ["A human individual", "A company or institution"],
            ],
            ["person", "person"],
        ),
        (
            [
                {"person": "A human individual"},
                {"location": "A geographical place"},
            ],
            [[["A human individual"]], [["A geographical place"]]],
            ["person", "location"],
        ),
        (
            ["person", "organization"],
            [["person", "organization"], ["person", "organization"]],
            ["person", "person"],
        ),
    ],
)
def test_inference_uses_description_prompts_but_returns_public_names(labels, expected_prompts, expected_output_names):
    model, collator = _inference_model()

    results = model.inference(
        ["Alice", "Paris"],
        labels,
        batch_size=1,
        return_class_probs=True,
    )

    assert collator.entity_types == expected_prompts
    assert [entities[0]["label"] for entities in results] == expected_output_names
    assert [list(entities[0]["class_probs"]) for entities in results] == [[name] for name in expected_output_names]
