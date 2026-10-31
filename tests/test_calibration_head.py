"""Tests for the GLiNER calibration head (gliner/modeling/calibrator.py) and its fitting utilities."""

from types import SimpleNamespace

import torch
import pytest

from gliner.calibration import (
    CachedOutput,
    balance_table,
    fit_calibrator,
    head_loss,
    span_reliability,
    assign_label_sets,
    candidate_cells,
    candidate_entries,
)
from gliner.modeling.calibrator import GLiNERCalibrator, apply_calibrator

D = 16


def _span_inputs(B=2, L=5, K=3, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, L, K, C, generator=g) * 3
    words = torch.randn(B, L, D, generator=g)
    prompts = torch.randn(B, C, D, generator=g)
    word_mask = torch.ones(B, L, dtype=torch.bool)
    word_mask[1, -2:] = False  # sample 1 has 3 words
    class_mask = torch.ones(B, C, dtype=torch.bool)
    class_mask[1, -1] = False
    return logits, words, prompts, word_mask, class_mask


def test_starts_at_global_scaling_and_only_touches_candidates():
    cal = GLiNERCalibrator(D, D, mode="span", candidate_threshold=0.2, max_width=3).eval()
    cal.reset_to_global_scaling(0.5, -1.0)
    logits, words, prompts, word_mask, class_mask = _span_inputs()
    out = cal(logits, words, prompts, word_mask, class_mask)
    assert torch.allclose(out["beta"], torch.full_like(out["beta"], 0.5), atol=1e-4)
    assert torch.allclose(out["bias"], torch.full_like(out["bias"], -1.0))

    calibrated = cal.calibrate(logits, words, prompts, word_mask, class_mask)
    is_cand = torch.zeros_like(logits, dtype=torch.bool)
    is_cand[out["index"]] = True
    assert torch.equal(calibrated[~is_cand], logits[~is_cand])
    assert torch.allclose(calibrated[is_cand], 0.5 * logits[is_cand] - 1.0, atol=1e-4)


def test_candidates_respect_threshold_span_bounds_and_padded_classes():
    cal = GLiNERCalibrator(D, D, mode="span", candidate_threshold=0.2, max_width=3)
    logits, words, prompts, word_mask, class_mask = _span_inputs()
    b, l, k, c = cal.gather(logits, words, prompts, word_mask, class_mask)["index"]
    assert (torch.sigmoid(logits[b, l, k, c]) >= 0.2 - 1e-6).all()
    num_words = word_mask.sum(-1)
    assert (l + k < num_words[b]).all()
    assert not ((b == 1) & (c == 2)).any()

    # Every eligible pair is a candidate.
    expected = 0
    for bi in range(2):
        for li in range(5):
            for ki in range(3):
                for ci in range(3):
                    valid = li + ki < num_words[bi] and class_mask[bi, ci]
                    expected += bool(valid and torch.sigmoid(logits[bi, li, ki, ci]) >= 0.2)
    assert len(b) == expected


def test_explicit_span_mode_uses_span_idx():
    cal = GLiNERCalibrator(D, D, mode="span", candidate_threshold=0.0).eval()
    logits = torch.zeros(1, 3, 2)
    span_idx = torch.tensor([[[0, 1], [2, 2], [3, 4]]])  # last span runs past 3 words
    span_mask = torch.tensor([[True, True, True]])
    words, prompts = torch.randn(1, 3, D), torch.randn(1, 2, D)
    out = cal(
        logits,
        words,
        prompts,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        span_idx,
        span_mask,
    )
    assert sorted(set(out["index"][1].tolist())) == [0, 1]
    # Mean feature of span (0, 1) is the mean of words 0 and 1.
    first = (out["index"][1] == 0).nonzero()[0, 0]
    assert torch.allclose(out["word_feats"][first, 2 * D :], words[0, :2].mean(0), atol=1e-5)


def test_token_mode_scales_all_channels_together():
    cal = GLiNERCalibrator(D, D, mode="token", candidate_threshold=0.1).eval()
    cal.reset_to_global_scaling(2.0, 0.5)
    logits = torch.randn(2, 4, 3, 3) * 3
    words, prompts = torch.randn(2, 4, D), torch.randn(2, 3, D)
    word_mask = torch.ones(2, 4, dtype=torch.bool)
    class_mask = torch.ones(2, 3, dtype=torch.bool)
    out = cal(logits, words, prompts, word_mask, class_mask)
    assert out["z"].shape[1] == 3
    calibrated = cal.calibrate(logits, words, prompts, word_mask, class_mask)
    b, w, c = out["index"]
    assert torch.allclose(calibrated[b, w, c], 2.0 * logits[b, w, c] + 0.5, atol=1e-4)
    assert (torch.sigmoid(logits[b, w, c]).amax(-1) >= 0.1 - 1e-6).all()


def test_without_bias_decisions_never_flip():
    cal = GLiNERCalibrator(D, D, use_bias=False, candidate_threshold=0.0)
    for p in cal.parameters():
        torch.nn.init.normal_(p, std=1.0)
    logits, words, prompts, word_mask, class_mask = _span_inputs()
    calibrated = cal.calibrate(logits, words, prompts, word_mask, class_mask)
    assert torch.equal(calibrated > 0, logits > 0)


def test_apply_calibrator_keeps_raw_logits():
    cal = GLiNERCalibrator(D, D, candidate_threshold=0.0, max_width=3).eval()
    cal.reset_to_global_scaling(0.5, 0.0)
    logits, words, prompts, word_mask, class_mask = _span_inputs()
    output = SimpleNamespace(
        logits=logits,
        words_embedding=words,
        prompts_embedding=prompts,
        mask=word_mask,
        prompts_embedding_mask=class_mask,
        span_logits=None,
        span_idx=None,
        span_mask=None,
    )
    apply_calibrator(cal, output)
    assert output.uncalibrated_logits is logits
    assert not torch.equal(output.logits, logits)


def _cached(label_vecs, n=40, seed=0):
    """Two labels, both with confident raw logits; label 0 is always wrong, label 1 always right."""
    g = torch.Generator().manual_seed(seed)
    examples = []
    for _ in range(n):
        W, K = 4, 2
        logits = torch.full((W, K, 2), -10.0)
        targets = torch.zeros(W, K, 2)
        logits[0, 0, 0] = 3.0 + torch.rand(1, generator=g).item()  # wrong
        logits[1, 0, 1] = 3.0 + torch.rand(1, generator=g).item()  # right
        targets[1, 0, 1] = 1.0
        examples.append(CachedOutput(torch.randn(W, D, generator=g), label_vecs.clone(), logits, targets, "span"))
    return examples


def test_fit_learns_where_backbone_is_wrong():
    torch.manual_seed(0)
    labels = torch.randn(2, D) * 3
    train, held = _cached(labels, 60, seed=1), _cached(labels, 20, seed=2)
    cal = GLiNERCalibrator(D, D, candidate_threshold=0.01, dropout=0.0, max_width=2)
    fit_calibrator(
        cal,
        train,
        held,
        "cpu",
        base_beta=1.0,
        base_bias=0.0,
        num_epochs=30,
        lr=3e-3,
        patience=10,
        log_fn=lambda *_: None,
    )
    batch = held[0]
    out = cal(
        batch.logits[None],
        batch.words[None].float(),
        batch.prompts[None].float(),
        torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )
    probs = torch.sigmoid(cal.calibrated_candidate_logits(out))
    wrong = probs[out["index"][-1] == 0]
    right = probs[out["index"][-1] == 1]
    assert wrong.max() < 0.3 < 0.7 < right.min()


def test_head_loss_prior_is_zero_at_base():
    out = {"z": torch.tensor([1.0, -2.0]), "beta": torch.tensor([0.5, 0.5]), "bias": torch.tensor([0.2, 0.2])}
    targets = torch.tensor([1.0, 0.0])
    loss, nll = head_loss(out, targets, base_beta=0.5, base_bias=0.2)
    assert torch.allclose(loss, nll)
    loss2, _ = head_loss(out, targets, base_beta=1.0, base_bias=0.0)
    assert loss2 > loss


def test_balance_table_equalises_cell_mass():
    torch.manual_seed(0)
    z = torch.randn(5000) * 3
    y = torch.bernoulli(torch.sigmoid(z / 3))
    table = balance_table(z, y, num_bins=5, max_weight=1e9)
    cells = candidate_cells(z, y, 5)
    mass = torch.zeros(10).index_add_(0, cells, table[cells])
    nonzero = mass[mass > 0]
    assert torch.allclose(nonzero, nonzero.mean().expand_as(nonzero), rtol=1e-4)
    assert torch.isclose(mass.sum(), torch.tensor(5000.0))


def test_candidate_entries_token_layout():
    cal = GLiNERCalibrator(D, D, mode="token", candidate_threshold=0.5)
    logits = torch.full((3, 2, 3), -5.0)
    logits[1, 0] = torch.tensor([2.0, 2.0, 2.0])
    targets = torch.zeros(3, 2, 3)
    targets[1, 0] = 1.0
    ex = CachedOutput(torch.randn(3, D), torch.randn(2, D), logits, targets, "token")
    z, y = candidate_entries(cal, [ex], "cpu")
    assert z.shape == (1, 3) and torch.equal(y, torch.ones(1, 3))


def test_assign_label_sets_deterministic_and_contains_gold():
    data = [{"tokenized_text": ["a"], "ner": [[0, 0, "x"]]}, {"tokenized_text": ["b"], "ner": []}]
    pool = ["x", "y", "z", "w"]
    a = assign_label_sets(data, "example", pool, seed=3)
    b = assign_label_sets(data, "example", pool, seed=3)
    assert a == b
    assert "x" in a[0]["ner_labels"] and len(a[0]["ner_labels"]) == 2
    assert len(a[1]["ner_labels"]) == 1
    assert assign_label_sets(data, "dataset", pool)[1]["ner_labels"] == pool
    kept = assign_label_sets([{**data[0], "ner_labels": ["q"]}], "example", pool)
    assert kept[0]["ner_labels"] == ["q"]


def test_span_reliability_bins_and_correctness():
    data = [
        {
            "tokenized_text": ["Barack", "Obama", "in", "Paris"],
            "ner": [[0, 1, "person"], [3, 3, "city"]],
            "ner_labels": ["person", "city"],
        }
    ]

    class FakeModel:
        def inference(self, texts, labels, **kwargs):
            assert texts == ["Barack Obama in Paris"]
            return [
                [
                    {"start": 0, "end": 12, "label": "person", "score": 0.95},  # correct
                    {"start": 16, "end": 21, "label": "person", "score": 0.92},  # wrong label
                    {"start": 16, "end": 21, "label": "city", "score": 0.35},  # correct
                ]
            ]

    r = span_reliability(FakeModel(), data, n_bins=10)
    assert r["num_spans"] == 3
    top = r["bins"][9]
    assert top["count"] == 2 and top["accuracy"] == pytest.approx(0.5)
    assert r["bins"][3]["accuracy"] == pytest.approx(1.0)
    expected = 2 / 3 * abs(0.935 - 0.5) + 1 / 3 * abs(0.35 - 1.0)
    assert r["ece"] == pytest.approx(expected)
    assert r["precision@0.5"] == pytest.approx(0.5)
    assert r["recall@0.5"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GLiNER integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_gliner():
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_config(
        {"model_name": "bert-base-uncased", "hidden_size": 32, "max_width": 4, "max_len": 64},
    )
    return model.to(torch.float32).eval()


def test_gliner_hook_calibrates_inference_and_round_trips(tiny_gliner, tmp_path):
    from gliner import GLiNER  # noqa: PLC0415

    model = tiny_gliner
    text, labels = "Barack Obama visited Paris", ["person", "city"]
    collator = model.create_collator()
    batch = collator([{"tokenized_text": text.split(), "ner": []}], entity_types=labels)
    raw = model.run_batch(batch).logits.clone()

    calibrator = model.add_calibrator(candidate_threshold=0.0)
    assert model.config.use_calibrator and model.calibrator is calibrator
    calibrator.reset_to_global_scaling(0.5, 0.25)
    out = model.run_batch(batch)
    assert torch.allclose(out.uncalibrated_logits, raw)
    valid = torch.isfinite(raw) & (raw > -1e3)
    changed = ~torch.isclose(out.logits, raw)
    assert changed.any() and (changed <= valid).all()

    with model.calibrator_disabled():
        assert torch.allclose(model.run_batch(batch).logits, raw)

    # With labels (training / loss computation) the hook is a no-op.
    train_batch = model._create_data_collator()([{"tokenized_text": text.split(), "ner": [[0, 1, "person"]]}])
    train_out = model(**train_batch)
    assert train_out.uncalibrated_logits is None

    model.save_pretrained(tmp_path)
    loaded = GLiNER.from_pretrained(str(tmp_path)).to(torch.float32).eval()
    assert type(loaded.calibrator).__name__ == "GLiNERCalibrator"
    assert torch.allclose(loaded.run_batch(batch).logits, out.logits, atol=1e-5)

    model.remove_calibrator()
    assert model.calibrator is None and not model.config.use_calibrator
    assert torch.allclose(model.run_batch(batch).logits, raw)


def test_add_calibrator_rejects_unknown_options(tiny_gliner):
    with pytest.raises(TypeError):
        tiny_gliner.add_calibrator(not_an_option=1)
    tiny_gliner.remove_calibrator()
