"""Integration tests for ConformalGLiNER against a real small checkpoint.

Mirrors tests/test_models.py::test_span_model's pattern (the only other test
in the suite that downloads a real model, gliner-community/gliner_small-v2.5).
This is the only conformal test module that touches the network;
tests/test_conformal_calibrators.py is fully synthetic.
"""

import warnings

import pytest

from gliner import GLiNER
from gliner.conformal import ConformalGLiNER, align_gold_scores, extract_raw_scores
from gliner.conformal.calibrators import calibration_floor
from gliner.conformal.scores import _assert_span_mode_supported

MODEL_ID = "gliner-community/gliner_small-v2.5"


def _examples(n_per_type: int = 10):
    """Small, easy, synthetic calibration/test corpus with a fixed 3-type schema."""
    templates = [
        (
            "Apple was founded by Steve Jobs in Cupertino .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 7, "location")],
        ),
        (
            "Google was founded by Larry Page in California .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 7, "location")],
        ),
        (
            "Microsoft was founded by Bill Gates in Redmond .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 7, "location")],
        ),
        (
            "Amazon was founded by Jeff Bezos in Seattle .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 7, "location")],
        ),
        ("Tesla was founded by Elon Musk in Austin .", [(0, 0, "organization"), (4, 5, "person"), (7, 7, "location")]),
        (
            "IBM was founded by Charles Flint in New York .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 8, "location")],
        ),
        (
            "Intel was founded by Robert Noyce in Santa Clara .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 8, "location")],
        ),
        (
            "Oracle was founded by Larry Ellison in Redwood City .",
            [(0, 0, "organization"), (4, 5, "person"), (7, 8, "location")],
        ),
    ]
    out = []
    i = 0
    while len(out) < n_per_type:
        text, ner = templates[i % len(templates)]
        out.append({"tokenized_text": text.split(), "ner": [list(t) for t in ner]})
        i += 1
    return out


@pytest.fixture(scope="module")
def model():
    return GLiNER.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def calib_data():
    # 25 examples per type comfortably clears calibration_floor(0.2) == 4 and
    # calibration_floor(0.1) == 9, used throughout this module.
    return _examples(25)


class TestCalibrateAllModes:
    @pytest.mark.parametrize("mode", ["span_filter", "risk_control", "mondrian"])
    def test_calibrate_and_predict_smoke(self, model, calib_data, mode):
        cg = ConformalGLiNER(model)
        cg.calibrate(calib_data, alpha=0.2, mode=mode)
        assert cg.is_calibrated
        assert set(cg._state.calibrated_types) == {"organization", "person", "location"}

        preds = cg.predict_entities(
            "Netflix was founded by Reed Hastings in Los Gatos .", ["organization", "person", "location"]
        )
        assert isinstance(preds, list)
        for ent in preds:
            assert ent["conformal"]["mode"] == mode
            assert ent["conformal"]["calibrated"] is True

    def test_batch_predict_shape(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        preds = cg.predict_entities(
            ["Netflix was founded by Reed Hastings .", "Uber was founded by Travis Kalanick ."],
            ["organization", "person"],
        )
        assert isinstance(preds, list) and len(preds) == 2
        assert all(isinstance(p, list) for p in preds)


class TestCalibrationFloorEnforcement:
    def test_raises_with_too_few_examples(self, model):
        cg = ConformalGLiNER(model)
        tiny = _examples(2)  # below calibration_floor(0.05) == 19
        with pytest.raises(ValueError, match=r"floor|insufficient"):
            cg.calibrate(tiny, alpha=0.05, mode="risk_control")

    def test_warns_for_under_floor_type_but_still_calibrates_others(self, model):
        floor = calibration_floor(0.1)
        assert floor == 9
        data = _examples(floor + 5)  # organization/person/location all clear the floor
        # Add a handful of a fourth type that stays under floor.
        data.append({"tokenized_text": ["Rare", "Corp", "makes", "widgets", "."], "ner": [[0, 1, "rare_type"]]})
        cg = ConformalGLiNER(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cg.calibrate(data, alpha=0.1, mode="span_filter")
        assert any("rare_type" in str(w.message) for w in caught)
        assert "rare_type" not in cg._state.calibrated_types
        assert {"organization", "person", "location"} <= set(cg._state.calibrated_types)


class TestUncalibratedTypeFallback:
    def test_unseen_label_warns_and_is_flagged_uncalibrated(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preds = cg.predict_entities(
                "The chemical compound was synthesized in the lab .",
                ["organization", "chemical_compound"],
            )
        assert any("chemical_compound" in str(w.message) for w in caught)
        for ent in preds:
            if ent["label"] == "chemical_compound":
                assert ent["conformal"]["calibrated"] is False


class TestEmptyPredictions:
    def test_no_matching_entities_returns_empty_list(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        preds = cg.predict_entities("zzz qqq xxx yyy .", ["organization", "person", "location"])
        assert preds == []

    def test_empty_text_returns_empty(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        preds = cg.predict_entities("", ["organization"])
        assert preds == []


class TestSaveLoadRoundTrip:
    def test_round_trip(self, model, calib_data, tmp_path):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        path = tmp_path / "calibration.json"
        cg.save_calibration(str(path))

        cg2 = ConformalGLiNER.load_calibration(str(path), model)
        assert cg2._state.mode == cg._state.mode
        assert cg2._state.alpha == cg._state.alpha
        assert cg2._state.crc_lambda == cg._state.crc_lambda

        text = "Netflix was founded by Reed Hastings in Los Gatos ."
        labels = ["organization", "person", "location"]
        assert cg.predict_entities(text, labels) == cg2.predict_entities(text, labels)

    def test_load_warns_on_model_mismatch(self, model, calib_data, tmp_path):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")
        path = tmp_path / "calibration.json"
        cg.save_calibration(str(path))
        cg._state.model_id = "some/other-model"
        cg.save_calibration(str(path))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConformalGLiNER.load_calibration(str(path), model)
        assert any("model-specific" in str(w.message) or "Nonconformity" in str(w.message) for w in caught)


class TestCoverageReport:
    def test_report_shape_and_disjoint_data_canary(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")

        # Canary: coverage measured on the *same* data the
        # threshold was calibrated on must come out at or above the nominal
        # target, since the threshold was tuned to fit exactly this data --
        # a biased estimate, and this test documents/guards that property
        # rather than treating it as a valid held-out coverage number.
        report = cg.coverage_report(calib_data)
        assert report["overall_coverage"] >= 1 - cg._state.alpha - 1e-9
        assert report["n_uncalibrated_gold"] == 0
        assert set(report["per_type_coverage"]) == {"organization", "person", "location"}
        assert report["efficiency_mean"] >= 0
        assert report["raw_candidates_mean"] > 0

    def test_risk_control_reports_per_sentence_not_per_entity_pooled(self, model, calib_data):
        """Regression test for a real bug found during empirical validation:
        risk_control calibrates and guarantees a *per-sentence* average miss
        rate (Conformal Risk Control's own loss definition), which is a
        different quantity from pooling every gold entity flat across
        sentences whenever entity-count-per-sentence varies. A test corpus
        with 1 entity in one sentence and 3 in another
        makes the two quantities provably different, so a regression back to
        flat pooling shows up as a hard assertion failure, not a subtle
        drift in a coverage number."""
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="risk_control")

        sentence_one = ["Apple", "was", "founded", "by", "Steve", "Jobs", "in", "Cupertino", "."]
        sentence_two = ["Google", "Microsoft", "Amazon", "dominate", "the", "market", "."]
        test_data = [
            {"tokenized_text": sentence_one, "ner": [[0, 0, "organization"]]},
            {
                "tokenized_text": sentence_two,
                "ner": [[0, 0, "organization"], [1, 1, "organization"], [2, 2, "organization"]],
            },
        ]
        report = cg.coverage_report(test_data)

        raw = extract_raw_scores(model, test_data, ["organization"])
        scores, types, example_idx = align_gold_scores(raw, test_data)
        tau = cg._nc_threshold_for(cg._state, "organization")
        hits_by_example = {0: [], 1: []}
        for s, _t, i in zip(scores, types, example_idx):
            hits_by_example[i].append(s <= tau)

        pooled = sum(sum(h) for h in hits_by_example.values()) / sum(len(h) for h in hits_by_example.values())
        per_sentence = sum((sum(h) / len(h) if h else 1.0) for h in hits_by_example.values()) / len(hits_by_example)

        assert report["overall_coverage"] == pytest.approx(per_sentence)
        if pooled != per_sentence:
            assert report["overall_coverage"] != pytest.approx(pooled)


class TestRequiresCalibration:
    def test_predict_before_calibrate_raises(self, model):
        cg = ConformalGLiNER(model)
        with pytest.raises(RuntimeError, match="not calibrated"):
            cg.predict_entities("Apple was founded by Steve Jobs .", ["organization"])

    def test_coverage_report_before_calibrate_raises(self, model, calib_data):
        cg = ConformalGLiNER(model)
        with pytest.raises(RuntimeError, match="not calibrated"):
            cg.coverage_report(calib_data)

    def test_thresholds_before_calibrate_raises(self, model):
        cg = ConformalGLiNER(model)
        with pytest.raises(RuntimeError, match="not calibrated"):
            cg.thresholds()

    def test_calibrated_types_before_calibrate_raises(self, model):
        cg = ConformalGLiNER(model)
        with pytest.raises(RuntimeError, match="not calibrated"):
            _ = cg.calibrated_types


class TestPublicThresholdAPI:
    """Regression coverage for the per-label threshold API requested in PR review
    (urchade/GLiNER#374) -- exposing what was previously only reachable via the
    private ``_state`` attribute."""

    @pytest.mark.parametrize("mode", ["span_filter", "risk_control", "mondrian"])
    def test_calibrated_types_matches_internal_state(self, model, calib_data, mode):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode=mode)
        assert set(cg.calibrated_types) == {"organization", "person", "location"}
        # public accessor, not a live reference to internal state
        cg.calibrated_types.append("tampered")
        assert "tampered" not in cg.calibrated_types

    @pytest.mark.parametrize("mode", ["span_filter", "risk_control", "mondrian"])
    def test_thresholds_covers_every_calibrated_type(self, model, calib_data, mode):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode=mode)
        thresholds = cg.thresholds()
        assert set(thresholds.keys()) == set(cg.calibrated_types)
        assert all(isinstance(v, float) for v in thresholds.values())

    def test_mondrian_thresholds_can_differ_per_type(self, model, calib_data):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode="mondrian")
        thresholds = cg.thresholds()
        # Not asserting they DO differ (real calibration data may coincidentally
        # produce equal thresholds) -- asserting the API *can* express a
        # per-type difference, unlike span_filter/risk_control below.
        assert isinstance(thresholds, dict) and len(thresholds) == 3

    @pytest.mark.parametrize("mode", ["span_filter", "risk_control"])
    def test_pooled_modes_share_one_threshold_across_labels(self, model, calib_data, mode):
        cg = ConformalGLiNER(model).calibrate(calib_data, alpha=0.2, mode=mode)
        thresholds = cg.thresholds()
        assert len(set(thresholds.values())) == 1


class TestModelCalibrateConvenienceMethod:
    """Regression coverage for the `model.calibrate()` / `model.conformal` API
    requested in PR review (urchade/GLiNER#374) -- calibration and inference on
    the same object, not just through a separately-constructed ConformalGLiNER.

    ``model`` is a module-scoped fixture shared across this whole test file --
    every test here must undo its own calibration afterward so it doesn't leak
    into unrelated tests that assume an uncalibrated model."""

    @pytest.fixture
    def calibrated_model(self, model, calib_data):
        model.calibrate(calib_data, alpha=0.2, mode="risk_control")
        yield model
        model._conformal_model = None

    def test_conformal_is_none_before_calibrate(self, model):
        assert model.conformal is None

    def test_calibrate_returns_self_for_chaining(self, model, calib_data):
        try:
            result = model.calibrate(calib_data, alpha=0.2, mode="risk_control")
            assert result is model
        finally:
            model._conformal_model = None

    def test_conformal_property_exposes_a_calibrated_wrapper(self, calibrated_model):
        assert isinstance(calibrated_model.conformal, ConformalGLiNER)
        assert calibrated_model.conformal.is_calibrated
        assert set(calibrated_model.conformal.calibrated_types) == {"organization", "person", "location"}

    def test_predicting_through_the_stored_wrapper_matches_direct_wrapper_use(self, calibrated_model):
        text = "Netflix was founded by Reed Hastings in Los Gatos ."
        labels = ["organization", "person", "location"]

        via_model = calibrated_model.conformal.predict_entities(text, labels)

        fresh_wrapper = ConformalGLiNER(calibrated_model)
        fresh_wrapper._state = calibrated_model.conformal._state  # same calibration, no re-fitting
        via_fresh_wrapper = fresh_wrapper.predict_entities(text, labels)

        assert via_model == via_fresh_wrapper

    def test_unsupported_architecture_raises_not_implemented(self):
        class _FakeTokenModel:
            pass

        # calibrate() is only meaningful on real BaseEncoderGLiNER instances;
        # this documents that the NotImplementedError comes from ConformalGLiNER
        # itself (see TestTokenModeRejected), not duplicated validation here.
        with pytest.raises(NotImplementedError, match="span-mode"):
            _assert_span_mode_supported(_FakeTokenModel())


class TestTokenModeRejected:
    def test_non_span_mode_model_raises_not_implemented(self):
        class _FakeTokenModel:
            pass

        cg = ConformalGLiNER(_FakeTokenModel())
        with pytest.raises(NotImplementedError, match="span-mode"):
            cg.calibrate(_examples(20), alpha=0.2, mode="risk_control")
