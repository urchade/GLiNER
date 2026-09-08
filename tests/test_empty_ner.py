"""Regression coverage for training on empty annotations (issue #320)."""

from copy import deepcopy

import torch
import pytest
from tokenizers import Tokenizer
from transformers import BertConfig, PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers.processors import BertProcessing
from tokenizers.pre_tokenizers import WhitespaceSplit

from gliner.config import (
    BiEncoderSpanConfig,
    BiEncoderTokenConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanRelexConfig,
)
from gliner.modeling.base import (
    BiEncoderSpanModel,
    BiEncoderTokenModel,
    UniEncoderSpanModel,
    UniEncoderTokenModel,
    UniEncoderSpanRelexModel,
)
from gliner.data_processing.collator import (
    SpanDataCollator,
    TokenDataCollator,
    RelationExtractionSpanDataCollator,
)
from gliner.data_processing.processor import (
    BiEncoderSpanProcessor,
    BiEncoderTokenProcessor,
    UniEncoderSpanProcessor,
    UniEncoderTokenProcessor,
    RelationExtractionSpanProcessor,
)


@pytest.fixture(
    params=[
        (UniEncoderSpanConfig, UniEncoderSpanProcessor, SpanDataCollator, UniEncoderSpanModel),
        (UniEncoderTokenConfig, UniEncoderTokenProcessor, TokenDataCollator, UniEncoderTokenModel),
        (BiEncoderSpanConfig, BiEncoderSpanProcessor, SpanDataCollator, BiEncoderSpanModel),
        (BiEncoderTokenConfig, BiEncoderTokenProcessor, TokenDataCollator, BiEncoderTokenModel),
        (
            UniEncoderSpanRelexConfig,
            RelationExtractionSpanProcessor,
            RelationExtractionSpanDataCollator,
            UniEncoderSpanRelexModel,
        ),
    ],
    ids=["uni_span", "uni_token", "bi_span", "bi_token", "relex_span"],
)
def setup(request):
    config_cls, processor_cls, collator_cls, model_cls = request.param
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "<<ENT>>",
        "<<SEP>>",
        "<<REL>>",
        "The",
        "weather",
        "is",
        "nice",
        ".",
        "Alice",
        "person",
    ]
    vocab = {word: index for index, word in enumerate(vocabulary)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = BertProcessing(("[SEP]", vocab["[SEP]"]), ("[CLS]", vocab["[CLS]"]))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=64,
        additional_special_tokens=["<<ENT>>", "<<SEP>>", "<<REL>>"],
    )
    encoder_config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    ).to_dict()
    kwargs = {}
    if processor_cls in (BiEncoderSpanProcessor, BiEncoderTokenProcessor):
        kwargs = {"labels_encoder": "local-bert", "labels_encoder_config": deepcopy(encoder_config)}
    config = config_cls(
        model_name="local-bert",
        encoder_config=encoder_config,
        hidden_size=16,
        vocab_size=len(tokenizer),
        class_token_index=vocab["<<ENT>>"],
        rel_token_index=vocab["<<REL>>"],
        num_rnn_layers=0,
        max_width=2,
        max_neg_type_ratio=0,
        dropout=0.0,
        augment_data_prob=0.0,
        **kwargs,
    )
    args = [config, tokenizer, None]
    if kwargs:
        args.append(tokenizer)
    processor = processor_cls(*args)
    return config, processor, collator_cls(config, data_processor=processor), model_cls, tokenizer


@pytest.mark.parametrize("case", ["single", "all_empty", "mixed", "explicit_negatives", "empty_labels"])
def test_empty_examples_produce_training_gradients(setup, case):
    config, processor, collator, model_cls, tokenizer = setup
    examples = [{"tokenized_text": ["The", "weather", "is", "nice", "."], "ner": [], "relations": []}]
    if case == "all_empty":
        examples.append({"tokenized_text": ["nice", "."], "ner": [], "relations": []})
    elif case == "mixed":
        examples.append({"tokenized_text": ["Alice"], "ner": [[0, 0, "person"]], "relations": []})
    elif case == "explicit_negatives":
        examples[0]["ner_negatives"] = ["person"]
    elif case == "empty_labels":
        examples[0]["ner_labels"] = []
    original = deepcopy(examples)
    raw = processor.collate_raw_batch(examples)
    candidates = raw["classes_to_id"][0]
    bi_encoder = isinstance(processor, (BiEncoderSpanProcessor, BiEncoderTokenProcessor))
    expected = "person" if case == "explicit_negatives" or (case == "mixed" and bi_encoder) else ""
    assert candidates == {expected: 1}
    if isinstance(processor, RelationExtractionSpanProcessor):
        assert all(mapping == {} for mapping in raw["rel_class_to_ids"])

    batch = collator(examples)
    assert torch.count_nonzero(batch["labels"][0]) == 0
    if not bi_encoder and expected == "":
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
        assert tokens[:3] == ["[CLS]", "<<ENT>>", "<<SEP>>"]
        # The empty label must not shift or discard any text words.
        assert batch["words_mask"][0][batch["words_mask"][0] > 0].tolist() == [1, 2, 3, 4, 5]

    model = model_cls(config, from_pretrained=False)
    output = model(**batch)
    assert torch.isfinite(output.loss) and output.loss > 0
    output.logits.retain_grad()
    output.loss.backward()
    assert output.logits.grad[0].abs().sum() > 0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert examples == original


def test_explicit_empty_inference_labels_remain_empty(setup):
    _, processor, _, _, _ = setup
    example = {"tokenized_text": ["Alice"], "ner": None, "relations": []}
    raw = processor.collate_raw_batch([example], entity_types=[[]])
    assert raw["classes_to_id"] == [{}]
