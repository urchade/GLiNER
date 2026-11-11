"""Explicit training labels must keep the negatives the data format documents."""

from types import SimpleNamespace

from gliner.data_processing.processor import (
    BiEncoderSpanProcessor,
    UniEncoderSpanProcessor,
    RelationExtractionSpanProcessor,
)

DOCUMENTED = {
    "tokenized_text": ["Apple", "Inc.", "hired", "Tim", "Cook", "in", "1998"],
    "ner": [[0, 1, "organization"], [3, 4, "person"], [6, 6, "date"]],
    "ner_labels": ["organization", "person", "date", "location"],
    "ner_negatives": ["product", "event", "money"],
}


def _processor(cls):
    processor = cls.__new__(cls)
    processor.config = SimpleNamespace(max_neg_type_ratio=1, max_types=25)
    return processor


def test_uni_encoder_keeps_explicit_negatives_with_ner_labels():
    mapping, _ = _processor(UniEncoderSpanProcessor).batch_generate_class_mappings([DOCUMENTED])

    assert mapping[0] == {
        "organization": 1,
        "person": 2,
        "date": 3,
        "location": 4,
        "product": 5,
        "event": 6,
        "money": 7,
    }


def test_bi_encoder_uses_ner_labels_not_only_gold_types():
    mapping, _ = _processor(BiEncoderSpanProcessor).batch_generate_class_mappings(
        [{"ner": [(0, 0, "organization")], "ner_labels": ["organization", "person", "location"]}]
    )

    assert set(mapping[0]) == {"organization", "person", "location"}


def test_bi_encoder_keeps_explicit_negatives_alongside_ner_labels():
    mapping, _ = _processor(BiEncoderSpanProcessor).batch_generate_class_mappings([DOCUMENTED])

    assert "location" in mapping[0]
    assert {"product", "event", "money"} <= set(mapping[0])


def test_relex_keeps_explicit_entity_and_relation_negatives():
    example = {**DOCUMENTED, "relations": [], "rel_labels": ["works_at"], "rel_negatives": ["founded"]}
    ner_mapping, _, rel_mapping, _ = _processor(RelationExtractionSpanProcessor).batch_generate_class_mappings(
        [example]
    )

    assert {"location", "product", "event", "money"} <= set(ner_mapping[0])
    assert set(rel_mapping[0]) == {"works_at", "founded"}
