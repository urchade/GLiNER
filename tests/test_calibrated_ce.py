"""Tests for the calibrated cross-entropy loss and post-hoc logit calibration."""

import json
from unittest.mock import Mock

import torch
import pytest
import torch.nn.functional as F

from gliner.config import (
    BiEncoderSpanConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanRelexConfig,
    UniEncoderTokenRelexConfig,
    UniEncoderSpanDecoderConfig,
)
from gliner.calibration import (
    calibrated_sigmoid,
    calibration_metrics,
    fit_logit_calibration,
    get_logit_calibration,
    set_logit_calibration,
    apply_logit_calibration,
)
from gliner.modeling.base import (
    BiEncoderSpanModel,
    UniEncoderSpanModel,
    UniEncoderTokenModel,
    UniEncoderSpanRelexModel,
    UniEncoderTokenRelexModel,
    UniEncoderSpanDecoderModel,
)
from gliner.decoding.decoder import SpanDecoder, TokenDecoder
from gliner.modeling.loss_functions import calibrated_ce_with_logits

# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class TestCalibratedCEFunction:
    def test_matches_hand_computed_value(self):
        s = torch.tensor([[2.0, -1.0, 0.5]])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        loss = calibrated_ce_with_logits(s, y)
        # L_pos = softplus(0 - 2); L_neg = logsumexp([0, -1, 0.5])
        expected = F.softplus(torch.tensor(-2.0)) + torch.logsumexp(torch.tensor([0.0, -1.0, 0.5]), 0)
        assert loss.shape == (1,)
        assert torch.allclose(loss[0], expected)

    def test_single_class_equals_two_way_softmax_ce(self):
        s = torch.randn(64, 1)
        y = torch.randint(0, 2, (64, 1)).float()
        loss = calibrated_ce_with_logits(s, y)
        two_way = torch.cat([torch.zeros_like(s), s], dim=-1)  # [anchor, class]
        expected = F.cross_entropy(two_way, y[:, 0].long(), reduction="none")
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_negative_only_row_is_softmax_ce_on_anchor(self):
        s = torch.randn(8, 5)
        y = torch.zeros(8, 5)
        loss = calibrated_ce_with_logits(s, y)
        logits = torch.cat([torch.zeros(8, 1), s], dim=-1)
        expected = F.cross_entropy(logits, torch.zeros(8, dtype=torch.long), reduction="none")
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_padded_classes_do_not_change_loss(self):
        s = torch.randn(4, 3)
        y = torch.zeros(4, 3)
        y[0, 1] = 1.0
        base = calibrated_ce_with_logits(s, y, calib_weight=0.5)

        padded_s = torch.cat([s, torch.full((4, 2), 50.0)], dim=-1)
        padded_y = torch.cat([y, torch.zeros(4, 2)], dim=-1)
        mask = torch.tensor([1, 1, 1, 0, 0]).bool()
        padded = calibrated_ce_with_logits(padded_s, padded_y, valid_mask=mask, calib_weight=0.5)
        assert torch.allclose(base, padded, atol=1e-6)

    def test_ignore_index_excluded(self):
        s = torch.tensor([[1.0, 30.0]])
        y = torch.tensor([[1.0, -100.0]])
        expected = calibrated_ce_with_logits(s[:, :1], y[:, :1])
        assert torch.allclose(calibrated_ce_with_logits(s, y), expected)

    def test_gradients_push_positives_up_and_top_negative_down(self):
        s = torch.tensor([[-0.5, 1.0, -3.0]], requires_grad=True)
        y = torch.tensor([[1.0, 0.0, 0.0]])
        calibrated_ce_with_logits(s, y).sum().backward()
        assert s.grad[0, 0] < 0  # positive is pushed up
        assert s.grad[0, 1] > 0  # strongest negative is pushed down
        assert s.grad[0, 1] > s.grad[0, 2] >= 0

    def test_calibration_term_adds_mean_bce(self):
        s = torch.randn(6, 4)
        y = (torch.rand(6, 4) > 0.7).float()
        plain = calibrated_ce_with_logits(s, y)
        calib = calibrated_ce_with_logits(s, y, calib_weight=0.3)
        bce = F.binary_cross_entropy_with_logits(s, y, reduction="none").mean(-1)
        assert torch.allclose(calib - plain, 0.3 * bce, atol=1e-6)

    def test_pos_weight_scales_positive_terms(self):
        s = torch.tensor([[-1.0, -2.0]])
        y = torch.tensor([[1.0, 0.0]])
        base = calibrated_ce_with_logits(s, y)
        weighted = calibrated_ce_with_logits(s, y, pos_weight=torch.tensor([[3.0, 3.0]]))
        assert torch.allclose(weighted - base, 2 * F.softplus(torch.tensor(1.0)))

    def test_fully_masked_rows_and_half_precision_are_finite(self):
        s = torch.randn(3, 4, dtype=torch.float16) * 20
        y = torch.zeros(3, 4, dtype=torch.float16)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        loss = calibrated_ce_with_logits(s, y, valid_mask=mask, calib_weight=0.5)
        assert torch.isfinite(loss).all()
        assert torch.allclose(loss, torch.zeros(3))

        loss = calibrated_ce_with_logits(s, y, calib_weight=0.5)
        assert torch.isfinite(loss).all()

    def test_training_learns_positive_above_anchor_above_negatives(self):
        torch.manual_seed(0)
        s = torch.zeros(32, 6, requires_grad=True)
        y = (torch.rand(32, 6) > 0.8).float()
        opt = torch.optim.SGD([s], lr=1.0)
        for _ in range(300):
            opt.zero_grad()
            calibrated_ce_with_logits(s, y, calib_weight=0.5).sum().backward()
            opt.step()
        assert (s[y > 0.5] > 0).all()
        assert (s[y < 0.5] < 0).all()


# ---------------------------------------------------------------------------
# Model losses
# ---------------------------------------------------------------------------


def _span_config(cls=UniEncoderSpanConfig, **extra):
    return cls(
        model_name="bert-base-uncased",
        hidden_size=64,
        dropout=0.1,
        max_width=12,
        span_mode="markerV0",
        class_token_index=103,
        post_fusion_schema="",
        embed_ent_token=True,
        **extra,
    )


def _span_tensors(B=2, L=6, K=12, C=4):
    torch.manual_seed(0)
    scores = torch.randn(B, L, K, C, requires_grad=True)
    labels = torch.zeros(B, L * K, C)
    labels[0, 3, 1] = 1.0
    labels[1, 7, 0] = 1.0
    labels[1, 7, 2] = 1.0
    prompts_mask = torch.ones(B, C)
    prompts_mask[1, -1] = 0
    span_mask = torch.ones(B, L * K)
    span_mask[1, -10:] = 0
    return scores, labels, prompts_mask, span_mask


@pytest.mark.parametrize("loss_type", ["ce", "ce_calibrated"])
@pytest.mark.parametrize("masking", ["none", "global", "label", "span"])
def test_span_model_ce_loss(loss_type, masking):
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    loss = model.loss(scores, labels, prompts_mask, span_mask, loss_type=loss_type, masking=masking, negatives=0.5)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_span_model_ce_ignores_padded_classes_and_spans():
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    base = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce_calibrated")

    noisy = scores.detach().clone().view(2, -1, 4)
    noisy[1, :, -1] = 100.0  # padded class
    noisy[1, -10:, :] = 100.0  # padded spans
    other = model.loss(noisy.view_as(scores), labels, prompts_mask, span_mask, loss_type="ce_calibrated")
    assert torch.allclose(base, other)


def test_span_model_ce_mean_reduction_divides_by_valid_spans():
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    total = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce", reduction="sum")
    mean = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce", reduction="mean")
    assert torch.allclose(mean, total / span_mask.sum())


def test_span_model_ce_with_span_width_weight():
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    plain = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce")
    weighted = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce", use_span_width_weight=True)
    assert weighted > plain


def test_span_model_ce_forward_with_kwargs():
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    B, L, W, C, K = 2, 20, 6, 3, 12
    inputs = {
        "input_ids": torch.randint(0, 1000, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "words_mask": torch.randint(0, W, (B, L)),
        "text_lengths": torch.tensor([W, W - 2]).unsqueeze(-1),
        "span_idx": torch.randint(0, W, (B, L * K, 2)),
        "span_mask": torch.randint(0, 2, (B, L * K)),
        "labels": torch.zeros(B, L * K, C),
    }
    out = model(**inputs, loss_type="ce_calibrated", ce_calib_weight=0.25)
    assert torch.isfinite(out.loss)


def test_bi_encoder_span_ce_loss():
    model = BiEncoderSpanModel(
        _span_config(BiEncoderSpanConfig, labels_encoder="bert-base-uncased"), from_pretrained=False
    )
    scores, labels, prompts_mask, span_mask = _span_tensors()
    loss = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce_calibrated")
    assert torch.isfinite(loss) and loss > 0


def test_span_decoder_ce_loss_combines_decoder_loss():
    config = _span_config(
        UniEncoderSpanDecoderConfig, labels_decoder="gpt2", decoder_mode="span", full_decoder_context=True
    )
    model = UniEncoderSpanDecoderModel(config, from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    span_only = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce")
    combined = model.loss(scores, labels, prompts_mask, span_mask, loss_type="ce", decoder_loss=torch.tensor(2.0))
    expected = 2.0 * config.decoder_loss_coef + span_only * config.span_loss_coef
    assert torch.allclose(combined, expected)


@pytest.mark.parametrize("last_dim", [3, None])
def test_token_model_ce_loss(last_dim):
    config = UniEncoderTokenConfig(
        model_name="bert-base-uncased", hidden_size=64, class_token_index=103, post_fusion_schema=""
    )
    model = UniEncoderTokenModel(config, from_pretrained=False)
    B, W, C = 2, 10, 4
    shape = (B, W, C, last_dim) if last_dim else (B, W, C)
    scores = torch.randn(*shape, requires_grad=True)
    labels = torch.zeros(*shape)
    labels[0, 2, 1] = 1.0
    prompts_mask = torch.ones(B, C)
    prompts_mask[0, -1] = 0
    word_mask = torch.ones(B, W)
    word_mask[1, -3:] = 0
    loss = model.loss(scores, labels, prompts_mask, word_mask, loss_type="ce_calibrated")
    assert torch.isfinite(loss) and loss > 0
    loss.backward()

    noisy = scores.detach().clone()
    noisy[0, :, -1] = 100.0
    noisy[1, -3:] = 100.0
    assert torch.allclose(loss, model.loss(noisy, labels, prompts_mask, word_mask, loss_type="ce_calibrated"))


def _relex_config(cls):
    return cls(
        model_name="bert-base-uncased",
        hidden_size=64,
        max_width=12,
        span_mode="markerV0",
        class_token_index=103,
        post_fusion_schema="",
        relations_layer="mlp",
        triples_layer=None,
        embed_rel_token=True,
        rel_token_index=104,
        rel_token="<<REL>>",
    )


def test_relex_relation_ce_loss():
    model = UniEncoderSpanRelexModel(_relex_config(UniEncoderSpanRelexConfig), from_pretrained=False)
    B, N, C = 2, 5, 3
    logits = torch.randn(B, N, C, requires_grad=True)
    labels = torch.zeros(B, N, C)
    labels[0, 1, 2] = 1.0
    pair_mask = torch.ones(B, N, C)
    pair_mask[1, -2:] = 0
    class_mask = torch.ones(B, N, C)
    loss = model.rel_loss(logits, labels, pair_mask, class_mask, loss_type="ce_calibrated")
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert logits.grad[1, -2:].abs().sum() == 0


def test_token_relex_ce_loss():
    model = UniEncoderTokenRelexModel(_relex_config(UniEncoderTokenRelexConfig), from_pretrained=False)
    B, W, C = 2, 8, 3
    scores = torch.randn(B, W, C, 3)
    labels = torch.zeros(B, W, C, 3)
    labels[0, 1, 0, 0] = 1.0
    loss = model.loss(scores, labels, torch.ones(B, C), torch.ones(B, W), loss_type="ce", span_mask=None)
    assert torch.isfinite(loss) and loss > 0


def test_focal_path_unchanged_by_default():
    model = UniEncoderSpanModel(_span_config(), from_pretrained=False)
    scores, labels, prompts_mask, span_mask = _span_tensors()
    focal = model.loss(scores, labels, prompts_mask, span_mask)
    expected = (
        F.binary_cross_entropy_with_logits(scores.view(2, -1, 4), labels, reduction="none")
        * prompts_mask[:, None]
        * span_mask[..., None]
    ).sum()
    assert torch.allclose(focal, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def test_training_args_accept_ce_and_reject_unknown(tmp_path):
    pytest.importorskip("accelerate")
    from gliner.training import TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path), loss_type="ce_calibrated", ce_calib_weight=0.25)
    assert args.loss_type == "ce_calibrated" and args.ce_calib_weight == 0.25
    with pytest.raises(ValueError, match="loss_type"):
        TrainingArguments(output_dir=str(tmp_path), loss_type="cross_entropy")


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------


class TestLogitCalibration:
    def test_defaults_are_identity(self):
        config = UniEncoderSpanConfig()
        logits = torch.randn(3, 4)
        assert get_logit_calibration(config) == (1.0, 0.0)
        assert apply_logit_calibration(logits, config) is logits
        # Mocks and objects without the attributes are treated as uncalibrated.
        assert apply_logit_calibration(logits, Mock()) is logits
        assert apply_logit_calibration(logits, None) is logits

    def test_temperature_and_bias_applied(self):
        config = UniEncoderSpanConfig(logit_temperature=2.0, logit_bias=-0.5, rel_logit_temperature=4.0)
        logits = torch.tensor([2.0, -2.0])
        assert torch.allclose(calibrated_sigmoid(logits, config), torch.sigmoid(logits / 2 - 0.5))
        assert torch.allclose(calibrated_sigmoid(logits, config, kind="relation"), torch.sigmoid(logits / 4))

    def test_invalid_temperature_rejected(self):
        with pytest.raises(ValueError):
            set_logit_calibration(UniEncoderSpanConfig(), temperature=0.0)

    def test_config_round_trip(self, tmp_path):
        config = UniEncoderSpanConfig()
        set_logit_calibration(config, 1.7, 0.2)
        path = tmp_path / "config.json"
        config.to_json_file(path)
        data = json.loads(path.read_text())
        assert data["logit_temperature"] == pytest.approx(1.7)
        assert UniEncoderSpanConfig(**data).logit_bias == pytest.approx(0.2)

    @pytest.mark.parametrize("fit_bias", [False, True])
    def test_fit_recovers_known_parameters(self, fit_bias):
        torch.manual_seed(0)
        true_t, true_b = 2.5, (-1.0 if fit_bias else 0.0)
        logits = torch.randn(200_000) * 4
        labels = torch.bernoulli(torch.sigmoid(logits / true_t + true_b))
        t, b = fit_logit_calibration(logits, labels, fit_bias=fit_bias)
        assert t == pytest.approx(true_t, rel=0.05)
        assert b == pytest.approx(true_b, abs=0.05)

        before = calibration_metrics(logits, labels)
        after = calibration_metrics(logits, labels, temperature=t, bias=b)
        assert after["nll"] < before["nll"]
        assert after["ece"] < before["ece"]

    def test_weighted_fit_matches_unweighted_on_duplicated_data(self):
        torch.manual_seed(1)
        logits = torch.randn(5000) * 3
        labels = torch.bernoulli(torch.sigmoid(logits / 2))
        w = torch.randint(1, 4, (5000,)).float()
        t_weighted, _ = fit_logit_calibration(logits, labels, weights=w)
        t_repeated, _ = fit_logit_calibration(logits.repeat_interleave(w.long()), labels.repeat_interleave(w.long()))
        assert t_weighted == pytest.approx(t_repeated, rel=1e-3)

    def test_span_decoder_uses_calibrated_scores(self):
        logits = torch.full((1, 3, 2, 1), -10.0)
        logits[0, 0, 0, 0] = 2.0
        tokens = [["a", "b", "c"]]
        raw = SpanDecoder(UniEncoderSpanConfig(max_width=2))
        calibrated = SpanDecoder(UniEncoderSpanConfig(max_width=2, logit_temperature=2.0))
        kwargs = dict(tokens=tokens, id_to_classes={1: "X"}, model_output=logits, threshold=0.5)
        (raw_span,) = raw.decode(**kwargs)[0]
        (cal_span,) = calibrated.decode(**kwargs)[0]
        assert raw_span.score == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-5)
        assert cal_span.score == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item(), abs=1e-5)

        # A bias moves the decision boundary.
        shifted = SpanDecoder(UniEncoderSpanConfig(max_width=2, logit_bias=-3.0))
        assert shifted.decode(**kwargs) == [[]]

    def test_token_decoder_uses_calibrated_scores(self):
        logits = torch.full((1, 3, 1, 3), -10.0)
        logits[0, 1, 0, :] = 3.0  # start, end, inside at word 1
        kwargs = dict(tokens=[["a", "b", "c"]], id_to_classes={1: "X"}, model_output=logits, threshold=0.5)
        raw = TokenDecoder(UniEncoderTokenConfig()).decode(**kwargs)[0]
        cal = TokenDecoder(UniEncoderTokenConfig(logit_temperature=3.0)).decode(**kwargs)[0]
        assert len(raw) == len(cal) == 1
        assert cal[0].score < raw[0].score
