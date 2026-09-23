"""Keep original annotation positions when input words produce no subtokens."""

from copy import deepcopy

import torch
import pytest
from tokenizers import Tokenizer
from transformers import BertConfig, PreTrainedTokenizerFast
from tokenizers.models import WordPiece
from tokenizers.processors import BertProcessing
from tokenizers.pre_tokenizers import WhitespaceSplit

from gliner.config import UniEncoderSpanConfig
from gliner.modeling.base import UniEncoderSpanModel
from gliner.modeling.utils import extract_word_embeddings
from gliner.data_processing.utils import prepare_word_mask
from gliner.data_processing.collator import SpanDataCollator
from gliner.data_processing.processor import UniEncoderSpanProcessor


@pytest.fixture
def tokenizer():
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "<<ENT>>",
        "<<SEP>>",
        "Organization",
        "Alice",
        "joined",
        "Ac",
        "##me",
        "yesterday",
    ]
    vocab = {word: index for index, word in enumerate(vocabulary)}
    backend = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = BertProcessing(("[SEP]", vocab["[SEP]"]), ("[CLS]", vocab["[CLS]"]))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=64,
        additional_special_tokens=["<<ENT>>", "<<SEP>>"],
    )


@pytest.mark.parametrize("invisible", ["", "\n", "\t", "  "])
@pytest.mark.parametrize(
    "position, expected",
    [(0, [2, 3]), (1, [1, 3]), (2, [1, 2])],
    ids=["leading", "interior", "trailing"],
)
def test_invisible_words_preserve_positions(tokenizer, invisible, position, expected):
    words = ["Alice", "joined"]
    words.insert(position, invisible)
    encoded = tokenizer([words], is_split_into_words=True)
    assert position not in encoded.word_ids(0)

    mask = prepare_word_mask([words], encoded)[0]

    assert [value for value in mask if value] == expected


@pytest.mark.parametrize("pooling", ["first", "last", "mean", "max"])
@pytest.mark.parametrize("token_level", [False, True])
@pytest.mark.parametrize("prompt_label", ["Organization", ""])
def test_subtokens_and_missing_prompt_words(tokenizer, pooling, token_level, prompt_label):
    prompt = ["<<ENT>>", prompt_label, "<<SEP>>"]
    texts = [[*prompt, "Alice", "\n", "Acme", "yesterday"]]
    encoded = tokenizer(texts, is_split_into_words=True)

    mask = prepare_word_mask(
        texts, encoded, skip_first_words=[len(prompt)], token_level=token_level, subtoken_pooling=pooling
    )[0]

    expected = [1, 3, 3, 4] if token_level or pooling in {"mean", "max"} else [1, 3, 4]
    assert [value for value in mask if value] == expected
    marked_tokens = [token for token, value in zip(encoded.tokens(0), mask, strict=True) if value == 3]
    expected_tokens = (
        ["Ac", "##me"] if token_level or pooling in {"mean", "max"} else ["Ac" if pooling == "first" else "##me"]
    )
    assert marked_tokens == expected_tokens


def test_different_prompt_lengths_and_padding(tokenizer):
    texts = [
        ["<<ENT>>", "Organization", "<<SEP>>", "\n", "", "Alice", "\t", "joined"],
        ["Alice", "joined"],
        ["<<ENT>>", "", "<<SEP>>", "\n", "\t"],
    ]
    encoded = tokenizer(texts, is_split_into_words=True, padding=True)

    masks = prepare_word_mask(texts, encoded, skip_first_words=[3, 0, 3])

    assert [[value for value in mask if value] for mask in masks] == [[3, 5], [1, 2], []]
    for mask, attention in zip(masks, encoded["attention_mask"], strict=True):
        assert len(mask) == len(attention)
        assert all(value == 0 for value, attended in zip(mask, attention, strict=True) if not attended)


@pytest.mark.parametrize("pooling", ["first", "last", "mean", "max"])
def test_span_labels_and_gradients_stay_on_annotated_word(tokenizer, pooling):
    encoder_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    ).to_dict()
    config = UniEncoderSpanConfig(
        model_name="local-bert",
        encoder_config=encoder_config,
        hidden_size=16,
        vocab_size=len(tokenizer),
        class_token_index=tokenizer.convert_tokens_to_ids("<<ENT>>"),
        num_rnn_layers=0,
        max_width=2,
        max_neg_type_ratio=0,
        dropout=0.0,
        augment_data_prob=0.0,
        subtoken_pooling=pooling,
    )
    processor = UniEncoderSpanProcessor(config, tokenizer, None)
    collator = SpanDataCollator(config, data_processor=processor)
    examples = [
        {
            "tokenized_text": ["Alice", "\n", "joined", "Acme", "yesterday"],
            "ner": [[3, 3, "Organization"]],
            "ner_labels": ["Organization"],
        }
    ]
    original = deepcopy(examples)
    batch = collator(examples)
    assert examples == original
    positive_spans = batch["span_idx"][0][batch["labels"][0, :, 0].bool()]
    assert positive_spans.tolist() == [[3, 3]]

    # Use distinct subtoken values to identify which word receives supervision.
    input_ids = batch["input_ids"]
    token_embeds = torch.arange(1, input_ids.numel() + 1, dtype=torch.float32).reshape(1, -1, 1)
    token_embeds.requires_grad_()
    words, _ = extract_word_embeddings(
        token_embeds,
        batch["words_mask"],
        batch["attention_mask"],
        batch_size=1,
        max_text_length=5,
        embed_dim=1,
        text_lengths=batch["text_lengths"],
        subtoken_pooling=pooling,
    )
    acme = (input_ids[0] == tokenizer.convert_tokens_to_ids("Ac")) | (
        input_ids[0] == tokenizer.convert_tokens_to_ids("##me")
    )
    values = token_embeds[0, acme, 0]
    expected = {"first": values[0], "last": values[-1], "mean": values.mean(), "max": values.max()}[pooling]
    torch.testing.assert_close(words[0, 3, 0], expected)
    assert words[0, 1, 0] == 0
    words[0, 3].sum().backward()
    assert token_embeds.grad[0, acme].abs().sum() > 0
    assert torch.count_nonzero(token_embeds.grad[0, ~acme]) == 0

    model = UniEncoderSpanModel(config, from_pretrained=False)
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
