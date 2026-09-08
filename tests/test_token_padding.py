from types import SimpleNamespace

import torch
import pytest

from gliner.model import UniEncoderTokenGLiNER
from gliner.decoding.decoder import TokenDecoder
from gliner.modeling.outputs import GLiNERBaseOutput
from gliner.data_processing.tokenizer import WordsSplitter


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.5, [0.2, 0.5, 0.2]])
@pytest.mark.parametrize("flat_ner", [True, False])
@pytest.mark.parametrize("padding_logit", [-1.0, 5.0])
def test_token_decoder_ignores_padding(threshold, flat_ner, padding_logit):
    decoder = TokenDecoder(SimpleNamespace())
    tokens = [["Alice"], ["works", "in", "Paris"], []]
    mappings = [{1: "person"}, {1: "location", 2: "organization"}, {1: "person"}]
    logits = torch.full((3, 3, 2, 3), -10.0)
    logits[0, 0, 0] = -0.5  # A valid low-confidence entity, retained at threshold 0.2.
    logits[0, 1:] = padding_logit
    logits[1, 2, 0] = 5.0
    logits[2] = padding_logit
    original_logits = logits.clone()

    results = decoder.decode(tokens, mappings, logits, threshold=threshold, flat_ner=flat_ner)

    for i, (row_tokens, spans) in enumerate(zip(tokens, results, strict=True)):
        row_threshold = threshold[i] if isinstance(threshold, list) else threshold
        unpadded = decoder.decode(
            [row_tokens],
            mappings[i],
            logits[i : i + 1, : len(row_tokens)],
            threshold=row_threshold,
            flat_ner=flat_ner,
        )[0]
        assert spans == unpadded
        assert all(0 <= span.start <= span.end < len(row_tokens) for span in spans)
    assert results[2] == []
    torch.testing.assert_close(logits, original_logits)


def test_low_threshold_token_inference_maps_only_real_words():
    model = UniEncoderTokenGLiNER.__new__(UniEncoderTokenGLiNER)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.data_processor = SimpleNamespace(words_splitter=WordsSplitter("whitespace"))
    model._inference_packing_config = None
    model.decoder = TokenDecoder(model.config)

    def collator(input_x, entity_types):
        return {
            "tokens": [item["tokenized_text"] for item in input_x],
            "id_to_classes": dict(enumerate(entity_types, start=1)),
        }

    def run_batch(batch, **kwargs):
        rows = batch["tokens"]
        logits = torch.full((len(rows), max(map(len, rows)), 1, 3), -10.0)
        for i, row in enumerate(rows):
            logits[i, len(row) - 1, 0] = -0.5
            logits[i, len(row) :] = -1.0  # Padding passes 0.2, but not 0.5.
        return GLiNERBaseOutput(logits=logits)

    model.create_collator = lambda: collator
    model.run_batch = run_batch
    texts = ["Alice", "", "works in Paris"]

    assert model.inference(texts, ["entity"], threshold=0.5) == [[], [], []]
    results = model.inference(texts, ["entity"], threshold=0.2)
    assert results == model.inference(texts, ["entity"], threshold=0.2, batch_size=1)
    assert [[entity["text"] for entity in entities] for entities in results] == [["Alice"], [], ["Paris"]]
    assert (results[0][0]["start"], results[0][0]["end"]) == (0, 5)
    assert (results[2][0]["start"], results[2][0]["end"]) == (9, 14)
