from unittest.mock import patch

import pytest
from tokenizers import Tokenizer
from transformers import GPT2Config, PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gliner.model import StreamingSpanGLiNER
from gliner.config import StreamingSpanConfig


@pytest.fixture
def model():
    words = ["[UNK]", "[PAD]", "Alice", "Paris", "works", "person", "location", "A", "human", "place"]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    config = StreamingSpanConfig(
        model_name="offline-test/missing-backbone",
        decoder_config=GPT2Config(
            vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, bos_token_id=0, eos_token_id=0
        ).to_dict(),
        labels_encoder_config={"model_type": "rnn"},
        hidden_size=16,
        max_width=2,
        num_rnn_layers=0,
        dropout=0.0,
    )
    model = StreamingSpanGLiNER(config, tokenizer=tokenizer)
    model._resize_token_embeddings(model, config, tokenizer)
    return model.eval()


@pytest.mark.parametrize(
    ("labels", "prompts", "names"),
    [
        ({"person": "A human", "location": "A place"}, [["A human", "A place"]] * 2, [["person", "location"]] * 2),
        ([{"person": "A human"}, {"location": "A place"}], [["A human"], ["A place"]], [["person"], ["location"]]),
        (["person", "person", "location"], [["person", "location"]] * 2, [["person", "location"]] * 2),
        ([["person"], ["location"]], [["person"], ["location"]], [["person"], ["location"]]),
        ("person", [["person"]] * 2, [["person"]] * 2),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sessions_preserve_prompts_and_public_names(model, labels, prompts, names, batch_size):
    for chunks, recompute in [(["Alice", "Paris"], False), ([" works", " works"], False), ([" Alice", " Paris"], True)]:
        with patch.object(model.data_processor, "collate_fn", wraps=model.data_processor.collate_fn) as collate:
            results = model.inference(
                chunks,
                labels,
                session_id=["a", "b"],
                batch_size=batch_size,
                threshold=0.0,
                return_class_probs=True,
                recompute=recompute,
            )
        encoded_prompts = [
            list(mapping) for call in collate.call_args_list for mapping in call.args[0]["classes_to_id"]
        ]
        assert encoded_prompts == prompts
        for session_id, entities, row_prompts, row_names in zip(["a", "b"], results, prompts, names, strict=True):
            state = model._session_cache.get(session_id)
            assert state.labels == tuple(row_prompts)
            assert state.label_names == tuple(row_names)
            assert entities
            assert all(entity["label"] in row_names for entity in entities)
            assert all(set(entity["class_probs"]) == set(row_names) for entity in entities)


def test_single_session_accepts_per_text_description_mapping(model):
    labels = [{"person": "A human"}]
    for chunk in ["Alice", " works"]:
        entities = model.inference([chunk], labels, session_id=["a"], threshold=0.0)[0]
        assert entities
        assert all(entity["label"] == "person" for entity in entities)
    assert model._session_cache.get("a").text == "Alice works"


@pytest.mark.parametrize("updated", [{"person": "A place"}, {"location": "A human"}])
def test_session_label_changes_require_recompute(model, updated):
    model.inference(["Alice"], {"person": "A human"}, session_id=["a"])
    with pytest.raises(ValueError, match="Labels for session 'a' changed"):
        model.inference([" works"], updated, session_id=["a"])
    assert model._session_cache.get("a").text == "Alice"

    results = model.inference([" works"], updated, session_id=["a"], recompute=True, threshold=0.0)
    assert results[0]
    assert all(entity["label"] in updated for entity in results[0])
    state = model._session_cache.get("a")
    assert state.labels == tuple(updated.values())
    assert state.label_names == tuple(updated)
    assert state.text == "Alice works"


def test_per_text_labels_stay_aligned_when_empty_chunks_are_skipped(model):
    results = model.inference(
        ["", "Paris"],
        [{"person": "A human"}, {"location": "A place"}],
        session_id=["empty", "city"],
        threshold=0.0,
    )
    assert results[0] == []
    assert results[1]
    assert all(entity["label"] == "location" for entity in results[1])
    assert model._session_cache.get("empty") is None
    assert model._session_cache.get("city").labels == ("A place",)


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        ([{"person": "A human"}], ValueError, "Per-text labels must have length 2"),
        ({"person": 1}, TypeError, "string keys and values"),
        ({"person": "same", "location": "same"}, ValueError, "descriptions must be unique"),
        ([{"person": "A human"}, {}], ValueError, "At least one label is required"),
    ],
)
def test_session_description_validation(model, labels, error, message):
    with pytest.raises(error, match=message):
        model.inference(["Alice", "Paris"], labels, session_id=["a", "b"])
    assert model.session_count == 0
