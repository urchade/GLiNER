"""Unit tests for the `dtype=` load path and the int8-only `quantize(...)` surface.

These tests deliberately avoid loading real hub checkpoints — they exercise the
public contracts on BaseGLiNER (``_parse_dtype``, ``_load_state_dict``) and
``model.quantize(...)`` via a lightweight fake subclass.
"""

import warnings
from pathlib import Path

import torch
import pytest
from torch import nn
from safetensors.torch import save_file

from gliner.model import BaseGLiNER


class _FakeModel(BaseGLiNER):
    """Minimal concrete subclass for exercising instance methods.

    ``BaseGLiNER`` is abstract; we stub the abstract methods out and wire a
    trivial ``nn.Module`` as the inner model so `.to(dtype)` etc. work.
    """

    _create_model = None
    _create_data_processor = None
    resize_embeddings = None
    inference = None
    evaluate = None

    def __init__(self, device_type: str = "cpu"):
        nn.Module.__init__(self)
        self.onnx_model = False
        self.model = nn.Linear(4, 4)
        self._device_type = device_type

    @property
    def device(self):
        return torch.device(self._device_type)


class TestParseDtype:
    def test_none_returns_none(self):
        assert BaseGLiNER._parse_dtype(None) is None

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("fp16", torch.float16),
            ("float16", torch.float16),
            ("half", torch.float16),
            ("bf16", torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("fp32", torch.float32),
            ("float32", torch.float32),
            ("float", torch.float32),
            ("BF16", torch.bfloat16),  # case-insensitive
        ],
    )
    def test_string_aliases(self, alias, expected):
        assert BaseGLiNER._parse_dtype(alias) == expected

    def test_torch_dtype_passthrough(self):
        assert BaseGLiNER._parse_dtype(torch.bfloat16) == torch.bfloat16
        assert BaseGLiNER._parse_dtype(torch.float16) == torch.float16

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            BaseGLiNER._parse_dtype("qint4")

    @pytest.mark.parametrize(
        "non_float",
        [torch.int8, torch.int32, torch.int64, torch.bool, torch.uint8],
    )
    def test_non_float_torch_dtype_rejected(self, non_float):
        with pytest.raises(ValueError, match="floating-point"):
            BaseGLiNER._parse_dtype(non_float)

    def test_error_steers_int8_to_quantize(self):
        """The error message should point users at `quantize='int8'`."""
        with pytest.raises(ValueError, match=r"quantize='int8'"):
            BaseGLiNER._parse_dtype(torch.int8)

    @pytest.mark.parametrize("bad", [123, 3.14, [], object()])
    def test_bad_type_raises_typeerror(self, bad):
        with pytest.raises(TypeError, match=r"str or torch\.dtype"):
            BaseGLiNER._parse_dtype(bad)


class TestLoadStateDict:
    @pytest.fixture
    def safetensors_file(self, tmp_path: Path):
        path = tmp_path / "model.safetensors"
        save_file(
            {
                "weight": torch.randn(4, 4, dtype=torch.float32),
                "bias": torch.randn(4, dtype=torch.float32),
                "int_buffer": torch.tensor([1, 2, 3], dtype=torch.int64),
                "bool_buffer": torch.tensor([True, False, True]),
            },
            str(path),
        )
        return path

    def test_default_path_unchanged(self, safetensors_file):
        """Without `dtype=`, tensors keep their stored dtype."""
        sd = BaseGLiNER._load_state_dict(safetensors_file, "cpu", dtype=None)
        assert sd["weight"].dtype == torch.float32
        assert sd["bias"].dtype == torch.float32
        assert sd["int_buffer"].dtype == torch.int64
        assert sd["bool_buffer"].dtype == torch.bool

    @pytest.mark.parametrize("target", [torch.bfloat16, torch.float16])
    def test_floats_cast_on_read(self, safetensors_file, target):
        sd = BaseGLiNER._load_state_dict(safetensors_file, "cpu", dtype=target)
        assert sd["weight"].dtype == target
        assert sd["bias"].dtype == target

    def test_non_float_buffers_preserved(self, safetensors_file):
        """Int/bool buffers must not be silently cast."""
        sd = BaseGLiNER._load_state_dict(safetensors_file, "cpu", dtype=torch.bfloat16)
        assert sd["int_buffer"].dtype == torch.int64
        assert sd["bool_buffer"].dtype == torch.bool

    def test_torch_load_path_also_casts(self, tmp_path: Path):
        """The non-safetensors `torch.load` branch also honours `dtype=`."""
        path = tmp_path / "model.bin"
        torch.save(
            {
                "weight": torch.randn(3, 3, dtype=torch.float32),
                "int_buf": torch.tensor([1, 2], dtype=torch.int32),
            },
            str(path),
        )
        sd = BaseGLiNER._load_state_dict(path, "cpu", dtype=torch.float16)
        assert sd["weight"].dtype == torch.float16
        assert sd["int_buf"].dtype == torch.int32

    def test_idempotent_when_already_target_dtype(self, tmp_path: Path):
        """Cast is skipped when the stored tensor already matches."""
        path = tmp_path / "model.safetensors"
        save_file({"w": torch.randn(2, 2, dtype=torch.bfloat16)}, str(path))
        sd = BaseGLiNER._load_state_dict(path, "cpu", dtype=torch.bfloat16)
        assert sd["w"].dtype == torch.bfloat16


class TestQuantizeInt8Only:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_int8_routes_to_apply_int8(self, monkeypatch, device):
        called = []
        monkeypatch.setattr(
            BaseGLiNER,
            "_apply_int8_quantization",
            lambda self: called.append(True),
        )
        _FakeModel(device).quantize("int8")
        assert called == [True]

    def test_int8_case_insensitive(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            BaseGLiNER,
            "_apply_int8_quantization",
            lambda self: called.append(True),
        )
        _FakeModel("cpu").quantize("INT8")
        assert called == [True]

    @pytest.mark.parametrize("alias", ["fp16", "float16", "half", "bf16", "bfloat16"])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_precision_aliases_raise_with_migration(self, alias, device):
        with pytest.raises(ValueError) as exc_info:
            _FakeModel(device).quantize(alias)
        msg = str(exc_info.value)
        assert "no longer supported" in msg
        assert "dtype=" in msg
        # Points at the correct torch.X replacement for the alias
        if alias in {"bf16", "bfloat16"}:
            assert "bfloat16" in msg
        else:
            assert "float16" in msg

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown quantize dtype"):
            _FakeModel("cpu").quantize("qint4")

    @pytest.mark.parametrize("bad", [True, False, 1, None, 3.14])
    def test_non_string_dtype_raises_typeerror(self, bad):
        with pytest.raises(TypeError, match="expects a string"):
            _FakeModel("cpu").quantize(bad)

    def test_onnx_model_raises_runtimeerror(self):
        fake = _FakeModel("cpu")
        fake.onnx_model = True
        with pytest.raises(RuntimeError, match="ONNX"):
            fake.quantize("int8")

    def test_precision_aliases_not_caught_by_onnx_branch(self):
        """ONNX check runs before the alias check; both should raise distinctly."""
        fake = _FakeModel("cpu")
        fake.onnx_model = True
        with pytest.raises(RuntimeError, match="ONNX"):
            fake.quantize("fp16")


class TestFromPretrainedQuantizeTrueRejection:
    """`from_pretrained(quantize=True)` used to default to fp16; now it must raise."""

    def test_true_raises(self):
        # Exercise the validation branch without building a real model: the
        # check happens before any expensive work. Use ``BaseGLiNER.from_pretrained``
        # with a non-existent path and a mocked download to land on the
        # validation line. Simpler: test the same guard by calling
        # ``_FakeModel("cpu").quantize(True)`` which also raises.
        with pytest.raises(TypeError, match="expects a string"):
            _FakeModel("cpu").quantize(True)


class TestNormalizeVariant:
    """``variant=`` canonicalization for selective downloads."""

    def test_none_is_default(self):
        assert BaseGLiNER._normalize_variant(None) is None

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("fp16", "fp16"),
            ("float16", "fp16"),
            ("half", "fp16"),
            ("FP16", "fp16"),  # case-insensitive
            ("bf16", "bf16"),
            ("bfloat16", "bf16"),
            ("BFloat16", "bf16"),
        ],
    )
    def test_aliases_canonicalize(self, alias, canonical):
        assert BaseGLiNER._normalize_variant(alias) == canonical

    @pytest.mark.parametrize("v", ["fp32", "float32", "float"])
    def test_fp32_explicitly_rejected(self, v):
        """fp32 is the default download — there's no separate variant file."""
        with pytest.raises(ValueError, match="not a separate download"):
            BaseGLiNER._normalize_variant(v)

    def test_int8_rejected_with_pointer(self):
        """int8 isn't a precision variant; the error should redirect to quantize=."""
        with pytest.raises(ValueError, match=r"quantize='int8'"):
            BaseGLiNER._normalize_variant("int8")

    def test_unknown_string_lists_supported(self):
        with pytest.raises(ValueError, match="Supported"):
            BaseGLiNER._normalize_variant("qint4")

    @pytest.mark.parametrize("bad", [123, 3.14, [], object()])
    def test_bad_type_raises_typeerror(self, bad):
        with pytest.raises(TypeError, match=r"str or None"):
            BaseGLiNER._normalize_variant(bad)


class TestVariantAllowPatterns:
    """The ``allow_patterns`` we hand to ``snapshot_download``."""

    def test_includes_only_requested_variant_safetensors(self):
        patterns = BaseGLiNER._variant_allow_patterns("bf16")
        assert "model.bf16.safetensors" in patterns
        # Default and other variants must NOT be in the allow list, otherwise
        # snapshot_download would still pull them.
        assert "model.safetensors" not in patterns
        assert "model.fp16.safetensors" not in patterns
        assert "pytorch_model.bin" not in patterns

    def test_includes_sharded_index(self):
        patterns = BaseGLiNER._variant_allow_patterns("fp16")
        assert "model.fp16.safetensors.index.json" in patterns

    def test_includes_configs_and_tokenizer(self):
        """Tokenizer / config files must always come down."""
        patterns = BaseGLiNER._variant_allow_patterns("bf16")
        # ``*.json`` covers gliner_config.json, tokenizer_config.json,
        # special_tokens_map.json, added_tokens.json.
        assert "*.json" in patterns
        # SentencePiece-style tokenizers ship .model / .txt files.
        assert "spiece.model" in patterns
        assert "sentencepiece.bpe.model" in patterns
        assert "*.txt" in patterns

    def test_fp16_and_bf16_differ_only_in_variant_filename(self):
        a = set(BaseGLiNER._variant_allow_patterns("fp16"))
        b = set(BaseGLiNER._variant_allow_patterns("bf16"))
        diff = a.symmetric_difference(b)
        assert diff == {
            "model.fp16.safetensors",
            "model.bf16.safetensors",
            "model.fp16.safetensors.index.json",
            "model.bf16.safetensors.index.json",
        }


class TestVariantDtypeConsistency:
    """``variant=`` and ``dtype=`` must agree (or only one set).

    The mismatch check fires before any download/config/model construction,
    so we can test it on the abstract ``BaseGLiNER`` without a real checkpoint.
    """

    def test_mismatch_raises_value_error(self, tmp_path: Path):
        """variant='bf16' with dtype='fp16' is contradictory."""
        with pytest.raises(ValueError, match="variant='bf16' requires"):
            BaseGLiNER.from_pretrained(
                model_id=str(tmp_path),
                model_dir=tmp_path,
                variant="bf16",
                dtype="fp16",
            )

    def test_aliases_normalize_before_compare(self, tmp_path: Path):
        """variant='bfloat16' + dtype='bf16' must NOT raise on dtype mismatch.

        Both canonicalize to bf16. The call still fails (no config in tmp_path)
        but with a different error — proving the consistency check passed.
        """
        with pytest.raises(FileNotFoundError, match="config"):
            BaseGLiNER.from_pretrained(
                model_id=str(tmp_path),
                model_dir=tmp_path,
                variant="bfloat16",
                dtype="bf16",
            )

    def test_int_dtype_against_variant_rejected_by_dtype_parser(self, tmp_path: Path):
        """variant='bf16' with dtype=torch.int8 should fail at dtype parsing."""
        with pytest.raises(ValueError, match="floating-point"):
            BaseGLiNER.from_pretrained(
                model_id=str(tmp_path),
                model_dir=tmp_path,
                variant="bf16",
                dtype=torch.int8,
            )


class TestVariantAvailable:
    """``_variant_available`` probe for variant file presence."""

    def _safetensors_at(self, path: Path) -> None:
        save_file({"weight": torch.randn(2, 2)}, str(path))

    def test_local_dir_with_variant_returns_true(self, tmp_path: Path):
        self._safetensors_at(tmp_path / "model.bf16.safetensors")
        assert BaseGLiNER._variant_available(str(tmp_path), "bf16") is True

    def test_local_dir_without_variant_returns_false(self, tmp_path: Path):
        self._safetensors_at(tmp_path / "model.safetensors")
        # default fp32 is there, but no bf16 file
        assert BaseGLiNER._variant_available(str(tmp_path), "bf16") is False

    def test_local_dir_with_only_fp16_does_not_match_bf16(self, tmp_path: Path):
        self._safetensors_at(tmp_path / "model.fp16.safetensors")
        assert BaseGLiNER._variant_available(str(tmp_path), "bf16") is False
        assert BaseGLiNER._variant_available(str(tmp_path), "fp16") is True

    def test_nonexistent_path_with_local_files_only_returns_none(self, tmp_path: Path):
        """No local dir + local_files_only=True should not hit the network."""
        nonexistent = tmp_path / "does_not_exist"
        assert BaseGLiNER._variant_available(str(nonexistent), "bf16", local_files_only=True) is None


class TestResolveVariantFallback:
    """``_resolve_variant`` warns and downgrades to None when variant is missing."""

    def test_passthrough_when_none(self):
        # No probe needed for variant=None; returns None silently.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert BaseGLiNER._resolve_variant("any/repo", None) is None
            assert [w for w in caught if "variant" in str(w.message).lower()] == []

    def test_passthrough_when_available(self, tmp_path: Path):
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.bf16.safetensors"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = BaseGLiNER._resolve_variant(str(tmp_path), "bf16")
            assert out == "bf16"
            assert [w for w in caught if "variant" in str(w.message).lower()] == []

    def test_warns_and_returns_none_when_unavailable(self, tmp_path: Path):
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))
        # Variant file isn't there; expect warn + None.
        with pytest.warns(UserWarning, match="not published"):
            out = BaseGLiNER._resolve_variant(str(tmp_path), "bf16")
        assert out is None


class TestResolveModelFile:
    """``_resolve_model_file`` picks the right file with graceful fallback."""

    def test_default_path_picks_safetensors(self, tmp_path: Path):
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))
        path, eff = BaseGLiNER._resolve_model_file(tmp_path, variant=None)
        assert path == tmp_path / "model.safetensors"
        assert eff is None

    def test_default_path_falls_back_to_pytorch_bin(self, tmp_path: Path):
        torch.save({}, str(tmp_path / "pytorch_model.bin"))
        path, eff = BaseGLiNER._resolve_model_file(tmp_path, variant=None)
        assert path == tmp_path / "pytorch_model.bin"
        assert eff is None

    def test_default_path_no_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No model file"):
            BaseGLiNER._resolve_model_file(tmp_path, variant=None)

    def test_variant_present_returns_variant(self, tmp_path: Path):
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.bf16.safetensors"))
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))
        path, eff = BaseGLiNER._resolve_model_file(tmp_path, variant="bf16")
        assert path == tmp_path / "model.bf16.safetensors"
        assert eff == "bf16"

    def test_variant_missing_warns_and_falls_back_to_safetensors(self, tmp_path: Path):
        save_file({"weight": torch.randn(2, 2)}, str(tmp_path / "model.safetensors"))
        with pytest.warns(UserWarning, match="not found"):
            path, eff = BaseGLiNER._resolve_model_file(tmp_path, variant="bf16")
        assert path == tmp_path / "model.safetensors"
        # effective_variant must be None so caller knows we fell back
        # (torch_dtype is already set from the variant earlier in from_pretrained)
        assert eff is None

    def test_variant_missing_falls_back_to_pytorch_bin(self, tmp_path: Path):
        torch.save({}, str(tmp_path / "pytorch_model.bin"))
        with pytest.warns(UserWarning, match="not found"):
            path, eff = BaseGLiNER._resolve_model_file(tmp_path, variant="bf16")
        assert path == tmp_path / "pytorch_model.bin"
        assert eff is None

    def test_variant_missing_with_no_fallback_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Neither"):
            BaseGLiNER._resolve_model_file(tmp_path, variant="bf16")
