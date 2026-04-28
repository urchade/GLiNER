import os
import re
import json
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Optional
from pathlib import Path

import torch
import onnxruntime as ort
import transformers
from tqdm import tqdm
from torch import nn
from packaging import version
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import (
    PyTorchModelHubMixin,
    hf_hub_download,
    snapshot_download,
    try_to_load_from_cache,
)
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from huggingface_hub.errors import EntryNotFoundError

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:
    quantize_dynamic, QuantType = None, None

from .utils import is_module_available
from .config import (
    GLiNERConfig,
    BaseGLiNERConfig,
    BiEncoderSpanConfig,
    BiEncoderTokenConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanRelexConfig,
    UniEncoderTokenRelexConfig,
    UniEncoderSpanDecoderConfig,
    UniEncoderTokenDecoderConfig,
)
from .decoding import (
    SpanDecoder,
    TokenDecoder,
    SpanRelexDecoder,
    TokenRelexDecoder,
    SpanGenerativeDecoder,
    TokenGenerativeDecoder,
)
from .training import Trainer, TrainingArguments
from .evaluation import BaseNEREvaluator, BaseRelexEvaluator
from .onnx.model import (
    BaseORTModel,
    BiEncoderSpanORTModel,
    BiEncoderTokenORTModel,
    UniEncoderSpanORTModel,
    UniEncoderTokenORTModel,
    UniEncoderSpanRelexORTModel,
    UniEncoderTokenRelexORTModel,
)
from .decoding.trie import LabelsTrie
from .infer_packing import InferencePackingConfig
from .modeling.base import (
    BaseModel,
    BiEncoderSpanModel,
    BiEncoderTokenModel,
    UniEncoderSpanModel,
    UniEncoderTokenModel,
    UniEncoderSpanRelexModel,
    UniEncoderTokenRelexModel,
    UniEncoderSpanDecoderModel,
    UniEncoderTokenDecoderModel,
)
from .modeling.utils import extract_prompt_features
from .data_processing import (
    BaseProcessor,
    BiEncoderSpanProcessor,
    BiEncoderTokenProcessor,
    UniEncoderSpanProcessor,
    UniEncoderTokenProcessor,
    UniEncoderSpanDecoderProcessor,
    RelationExtractionSpanProcessor,
    UniEncoderTokenDecoderProcessor,
    RelationExtractionTokenProcessor,
)
from .data_processing.collator import (
    BiEncoderSpanDataCollator,
    BiEncoderTokenDataCollator,
    UniEncoderSpanDataCollator,
    UniEncoderTokenDataCollator,
    UniEncoderSpanDecoderDataCollator,
    RelationExtractionSpanDataCollator,
    UniEncoderTokenDecoderDataCollator,
    RelationExtractionTokenDataCollator,
)
from .data_processing.tokenizer import WordsSplitter

if is_module_available("onnxruntime"):
    import onnxruntime as ort

    ONNX_AVAILABLE = True
else:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
    config_class: type = None
    model_class: type = None
    ort_model_class: type = None
    data_processor_class: type = None
    data_collator_class: type = None
    decoder_class: type = None

    def __init__(
        self,
        config: BaseGLiNERConfig,
        model: Optional[BaseModel] = None,
        tokenizer: Optional[BaseModel] = None,
        data_processor: Optional[BaseProcessor] = None,
        backbone_from_pretrained: Optional[bool] = False,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initialize a BaseGLiNER model.

        Args:
            config: Model configuration object.
            model: Pre-initialized model instance. If None, creates a new model.
            tokenizer: Pre-initialized tokenizer. If None, creates a new tokenizer.
            data_processor: Pre-initialized data processor. If None, creates a new processor.
            backbone_from_pretrained: Whether to load the backbone from pretrained weights.
            cache_dir: Directory for caching downloaded models.
            **kwargs: Additional keyword arguments passed to model creation.
        """
        super().__init__()
        self.config = config

        if model is not None:
            self.model = model
        else:
            self.model = self._create_model(config, backbone_from_pretrained, cache_dir, **kwargs)

        if data_processor is not None:
            self.data_processor = data_processor
        else:
            self.data_processor = self._create_data_processor(config, cache_dir, tokenizer, **kwargs)

        if isinstance(self.model, BaseORTModel):
            self.onnx_model = True
        else:
            self.onnx_model = False

        self.decoder = self.decoder_class(config)

        self._keys_to_ignore_on_save = None
        self._inference_packing_config: Optional[InferencePackingConfig] = None

    @abstractmethod
    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        """
        Create model instance. Must be implemented by child classes.

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def _create_data_processor(self, config, cache_dir, tokenizer=None, **kwargs):
        """
        Create data processor instance. Must be implemented by child classes.

        Returns:
            Data processor instance
        """
        pass

    @abstractmethod
    def resize_embeddings(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def forward(self, *args, **kwargs):
        """Forward pass through the model.

        Args:
            *args: Positional arguments passed to the model.
            **kwargs: Keyword arguments passed to the model.

        Returns:
            Model output from the forward pass.
        """
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        """Get the device where the model is located.

        Returns:
            Torch device object (CPU or CUDA).
        """
        if self.onnx_model:
            providers = self.model.session.get_providers()
            if "CUDAExecutionProvider" in providers:
                return torch.device("cuda")
            return torch.device("cpu")
        device = next(self.model.parameters()).device
        return device

    def configure_inference_packing(self, config: Optional[InferencePackingConfig]) -> None:
        """Configure default packing behavior for inference calls.

        Passing ``None`` disables packing by default. Individual inference
        methods accept a ``packing_config`` argument to override this setting
        on a per-call basis.

        Args:
            config: Inference packing configuration or None to disable packing.
        """
        self._inference_packing_config = config

    def compile(self):
        """Compile the model using torch.compile for optimization.

        Uses ``dynamic=True`` to generate shape-generic kernels, which avoids
        recompilation on variable-length NER inputs. Also enables
        ``capture_scalar_outputs`` to trace through data-dependent shape
        operations (e.g., computing max number of entity types per batch).

        Best combined with ``quantize()`` for maximum throughput (~1.9x over fp32).

        When FlashDeBERTa is active, its custom Triton kernels are incompatible
        with torch.compile tracing.  The encoder forward is automatically
        wrapped with ``torch.compiler.disable`` so the rest of the model
        (span representation, scoring, etc.) still benefits from compilation.
        """
        torch._dynamo.config.capture_scalar_outputs = True

        # FlashDeBERTa uses hand-written Triton kernels that torch.compile cannot trace.
        try:
            bert_layer = self.model.token_rep_layer.bert_layer
            model_cls = bert_layer.model.__class__.__name__
            if model_cls == "FlashDebertaV2Model":
                bert_layer.forward = torch.compiler.disable(bert_layer.forward)
        except AttributeError:
            pass  # non-standard architecture, skip

        self.model = torch.compile(self.model, dynamic=True)

    _PRECISION_ALIASES = {"fp16", "float16", "half", "bf16", "bfloat16"}

    _DTYPE_MAP = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    # Variants the loader knows how to *download* selectively. The default
    # (fp32) is represented as `variant=None` and matches the canonical
    # ``model.safetensors`` filename. fp16/bf16 map to the transformers
    # convention ``model.{variant}.safetensors``.
    _VARIANT_TO_DTYPE = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    _VARIANT_ALIASES = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }

    @classmethod
    def _normalize_variant(cls, variant) -> Optional[str]:
        """Canonicalize a variant string to ``"fp16"``, ``"bf16"``, or ``None``.

        ``None`` selects the default (fp32) ``model.safetensors``. Anything
        else is canonicalized via :attr:`_VARIANT_ALIASES`. Unknown values —
        including ``"fp32"`` and integer dtypes — raise ``ValueError`` with a
        message pointing the caller at ``dtype=`` for in-memory casts.
        """
        if variant is None:
            return None
        if not isinstance(variant, str):
            raise TypeError(f"variant must be str or None, got {type(variant).__name__}")
        key = variant.lower()
        if key in {"fp32", "float32", "float"}:
            raise ValueError(
                "variant='fp32' is not a separate download — drop variant= to load the default `model.safetensors`."
            )
        if key not in cls._VARIANT_ALIASES:
            raise ValueError(
                f"Unknown variant {variant!r}. Supported: {sorted(set(cls._VARIANT_ALIASES))}. "
                f"For int8, use `quantize='int8'` (no separate download)."
            )
        return cls._VARIANT_ALIASES[key]

    @staticmethod
    def _variant_allow_patterns(variant: str) -> list:
        """Return ``snapshot_download(allow_patterns=...)`` for a variant.

        Includes the single-file variant safetensors, the sharded variant
        index, the actual sharded variant safetensors files, and the configs
        and tokenizer assets every load needs. The default
        ``model.safetensors`` and ``pytorch_model.bin`` are deliberately
        excluded so the caller pays I/O only for the requested variant.

        Sharded checkpoint convention (transformers-style):
            ``model-00001-of-NNNNN.{variant}.safetensors``
            ``model.{variant}.safetensors.index.json``
        """
        return [
            "*.json",
            "*.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
            # Single-file variant.
            f"model.{variant}.safetensors",
            # Sharded variant: index file + per-shard files.
            f"model.{variant}.safetensors.index.json",
            f"model-*-of-*.{variant}.safetensors",
        ]

    @classmethod
    def _variant_available(
        cls,
        model_id: str,
        variant: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Union[str, bool, None] = None,
        local_files_only: bool = False,
    ) -> Optional[bool]:
        """Probe whether ``model.{variant}.safetensors`` is published.

        Resolution order (matches the ``transformers`` / ``huggingface_hub``
        idiom — cheapest checks first, no list-files round-trip):

        1. ``model_id`` is a local directory → ``Path.exists()``.
        2. The file is in the local HF cache for this repo+revision →
           ``try_to_load_from_cache``. Pure local lookup, no network.
        3. ``local_files_only=True`` → return ``None`` (uncertain) without
           hitting the network.
        4. ``hf_hub_download`` for the variant filename: success means the
           file exists (and is now cached, so the subsequent
           ``snapshot_download`` reuses it); ``EntryNotFoundError`` means
           the publisher hasn't uploaded a variant.

        Returns:
            ``True`` / ``False`` when known, or ``None`` when availability
            cannot be determined (offline + uncached, transient API failure,
            gated repo without auth). ``None`` is treated as "try the narrow
            download and let it fail loudly".
        """
        target = f"model.{variant}.safetensors"

        # 1. Local directory.
        model_dir = Path(model_id)
        if model_dir.exists() and model_dir.is_dir():
            return (model_dir / target).exists()

        # 2. Already cached for this repo+revision? Pure local — no network.
        # try_to_load_from_cache validates the repo_id format; an
        # HFValidationError here means the input isn't a valid repo_id at
        # all (e.g. a non-existent local path), so treat as uncertain.
        # ``cache_dir`` must match what ``snapshot_download`` will use, or the
        # probe and the actual download diverge (and we'd download the variant
        # twice).
        try:
            cached = try_to_load_from_cache(repo_id=model_id, filename=target, revision=revision, cache_dir=cache_dir)
        except Exception:
            return None
        if isinstance(cached, str):
            return True

        # 3. Offline mode: can't probe further.
        if local_files_only:
            return None

        # 4. Try-and-recover via hf_hub_download. Success caches the file so
        # the subsequent snapshot_download reuses it (no double download).
        # cache_dir must propagate so the probe and snapshot_download share
        # the same store.
        try:
            hf_hub_download(
                repo_id=model_id,
                filename=target,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )
            return True
        except EntryNotFoundError:
            return False
        except Exception:
            # Auth / network / repo-not-found / gated — surface as uncertain
            # and let the subsequent download path produce the canonical error.
            return None

    @classmethod
    def _resolve_variant(
        cls,
        model_id: str,
        variant: Optional[str],
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Union[str, bool, None] = None,
        local_files_only: bool = False,
    ) -> Optional[str]:
        """Probe for variant availability; warn and fall back to ``None`` if missing.

        When the publisher has uploaded ``model.{variant}.safetensors`` this
        returns ``variant`` unchanged so the caller proceeds with the narrow
        download (the I/O win). When the file is definitively absent, this
        returns ``None`` and emits a ``UserWarning`` — the caller should fall
        back to the default fp32 file plus an in-memory cast (via ``dtype=``).
        Uncertain probes (``available is None``) pass through unchanged so
        the narrow download is attempted.
        """
        if variant is None:
            return None
        available = cls._variant_available(
            model_id,
            variant,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        if available is False:
            # TODO(strict-variant): once half-precision variant files have been
            # published for the flagship GLiNER models on the Hub, flip this
            # branch to raise ``EntryNotFoundError`` (or a wrapped equivalent)
            # instead of warning + falling back. That matches
            # ``transformers.PreTrainedModel.from_pretrained(variant=...)``,
            # which is strict — explicit > implicit. The soft-fallback was a
            # transitional choice taken because, at PR-merge time, no GLiNER
            # repo on the Hub shipped a variant file, so a strict surface
            # would have been broken-on-arrival for every caller.
            warnings.warn(
                f"variant={variant!r} requested but 'model.{variant}.safetensors' is not "
                f"published in {model_id!r}. Falling back to the default fp32 file with "
                f"dtype={variant!r} cast on read — bytes-on-the-wire are not reduced. To "
                f"silence this, ask the publisher to upload the variant or pass variant=None.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return variant

    @classmethod
    def _resolve_model_file(cls, model_dir: Path, variant: Optional[str]) -> tuple:
        """Pick the model file on disk, with graceful fallback if variant is missing.

        Returns ``(model_file: Path, effective_variant: Optional[str])``.
        ``effective_variant`` is ``None`` when we fell back from a requested
        variant to the fp32 file — the caller already has ``torch_dtype`` set
        from the variant, so cast-on-read still produces the right precision.

        Raises:
            FileNotFoundError: if no usable model file exists.
        """
        if variant is not None:
            variant_file = model_dir / f"model.{variant}.safetensors"
            if variant_file.exists():
                return variant_file, variant
            # Variant requested but missing — try fp32 fallback alongside.
            fallback = model_dir / "model.safetensors"
            if not fallback.exists():
                fallback = model_dir / "pytorch_model.bin"
            if fallback.exists():
                warnings.warn(
                    f"Variant file 'model.{variant}.safetensors' not found in {model_dir}; "
                    f"loading '{fallback.name}' and casting to {variant} on read.",
                    UserWarning,
                    stacklevel=3,
                )
                return fallback, None
            raise FileNotFoundError(
                f"Neither 'model.{variant}.safetensors' nor 'model.safetensors' "
                f"nor 'pytorch_model.bin' found in {model_dir}."
            )
        # No variant requested — original behavior.
        model_file = model_dir / "model.safetensors"
        if not model_file.exists():
            model_file = model_dir / "pytorch_model.bin"
        if not model_file.exists():
            raise FileNotFoundError(f"No model file found in {model_dir}")
        return model_file, None

    @classmethod
    def _parse_dtype(cls, dtype) -> Optional[torch.dtype]:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            resolved = dtype
        elif isinstance(dtype, str):
            key = dtype.lower()
            if key not in cls._DTYPE_MAP:
                raise ValueError(f"Unknown dtype {dtype!r}. Supported: {sorted(cls._DTYPE_MAP.keys())}")
            resolved = cls._DTYPE_MAP[key]
        else:
            raise TypeError(f"dtype must be str or torch.dtype, got {type(dtype).__name__}")
        # ``instance.model.to(dtype)`` only accepts floating-point/complex dtypes;
        # reject e.g. torch.int8 / torch.bool up front with a clearer error than
        # the one PyTorch raises deep in the load path.
        if not resolved.is_floating_point:
            raise ValueError(
                f"dtype must be a floating-point dtype (e.g. torch.bfloat16, torch.float16, "
                f"torch.float32), got {resolved}. For int8 quantization use `quantize='int8'`."
            )
        return resolved

    def quantize(self, dtype: str = "int8") -> None:
        """Apply int8 quantization to the model.

        Only ``"int8"`` is accepted; for precision changes (fp16/bf16), use
        ``dtype=`` on :meth:`GLiNER.from_pretrained` or ``model.to(torch_dtype)``
        — those are downcasts, not quantization, and were removed from this API.

        Args:
            dtype: Must be ``"int8"``. On CPU, uses PyTorch's built-in dynamic
                quantization with FBGEMM int8 kernels (~1.6x speedup). On GPU,
                uses ``torchao`` int8 weight-only quantization (~50% memory
                reduction, no speed gain; requires the ``torchao`` package).
                Stock DeBERTa-based models lose accuracy with int8; use this
                with models fine-tuned with quantization-aware training (QAT).

        Raises:
            RuntimeError: If the model is an ONNX model (use ONNX quantization instead).
            ValueError: If *dtype* is not ``"int8"``. Precision aliases (fp16/bf16) raise
                with a migration message pointing at ``dtype=`` / ``model.to(...)``.
            ImportError: If ``torchao`` is not installed and int8 on GPU is requested.

        Examples:
            >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", map_location="cuda")
            >>> model.quantize("int8")  # int8 (torchao on GPU, FBGEMM on CPU)
            >>> # For precision-only changes, prefer:
            >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", dtype="bf16")
        """
        if self.onnx_model:
            raise RuntimeError(
                "Cannot apply PyTorch quantization to an ONNX model. "
                "Use export_to_onnx(quantize=True) for ONNX quantization."
            )

        if not isinstance(dtype, str):
            raise TypeError(
                f"`quantize()` expects a string (only 'int8' is supported), "
                f"got {type(dtype).__name__}. For precision changes use `dtype=` "
                f"on `GLiNER.from_pretrained` or `model.to(torch_dtype)`."
            )

        dtype_lower = dtype.lower()
        if dtype_lower in self._PRECISION_ALIASES:
            torch_name = "bfloat16" if dtype_lower in {"bf16", "bfloat16"} else "float16"
            raise ValueError(
                f"`quantize({dtype!r})` is no longer supported — these values were precision "
                f"downcasts, not quantization. Use `GLiNER.from_pretrained(..., dtype={dtype!r})` "
                f"for an efficient load, or `model.to(torch.{torch_name})` post-load. "
                f"`quantize('int8')` is the only remaining value."
            )
        if dtype_lower != "int8":
            raise ValueError(
                f"Unknown quantize dtype {dtype!r}. Only 'int8' is supported; "
                f"use `dtype=` on `GLiNER.from_pretrained` for precision changes."
            )
        self._apply_int8_quantization()

    def _apply_int8_quantization(self) -> None:
        """Apply int8 quantization using the best backend for the current device.

        - **CPU**: Uses ``torch.ao.quantization.quantize_dynamic`` with FBGEMM
          int8 kernels.  ~1.6x faster than fp32, no extra dependencies.
        - **GPU**: Uses ``torchao.quantize_`` with ``Int8WeightOnlyConfig``.
          Requires the ``torchao`` package.
        """
        if self.device.type == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            logger.info("Applied int8 dynamic quantization to all nn.Linear layers (CPU).")
        else:
            if is_module_available("torchao"):
                import torchao  # noqa: PLC0415
                from torchao.quantization import Int8WeightOnlyConfig  # noqa: PLC0415
            else:
                raise ImportError(
                    "int8 quantization on GPU requires the 'torchao' package. Install it with: pip install torchao"
                ) from None

            torchao.quantize_(self.model, Int8WeightOnlyConfig())
            logger.info("Applied int8 weight-only quantization via torchao (GPU).")

    def _get_special_tokens(self):
        """Get special tokens to add to tokenizer.

        Can be overridden by child classes.

        Returns:
            List of special tokens
        """
        tokens = ["[FLERT]", self.config.ent_token, self.config.sep_token]
        return tokens

    def prepare_state_dict(self, state_dict):
        """Prepare state dict for saving, handling torch.compile artifacts.

        Args:
            state_dict: Original state dictionary from the model.

        Returns:
            Cleaned state dictionary with torch.compile prefixes removed.
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            _key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[_key] = tensor
        return new_state_dict

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[BaseGLiNERConfig] = None,
        repo_id: Optional[str] = None,
        push_to_hub: bool = False,
        safe_serialization: bool = False,
        **push_to_hub_kwargs,
    ) -> Optional[str]:
        """Save model weights and configuration to local directory.

        Args:
            save_directory: Path to directory for saving.
            config: Model configuration. Uses self.config if None.
            repo_id: Repository ID for hub upload.
            push_to_hub: Whether to push to HuggingFace Hub.
            safe_serialization: Whether to use safetensors format.
            **push_to_hub_kwargs: Additional arguments for push_to_hub.

        Returns:
            Repository URL if pushed to hub, None otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_state_dict = self.prepare_state_dict(self.model.state_dict())

        if safe_serialization:
            save_file(model_state_dict, save_directory / "model.safetensors")
        else:
            torch.save(model_state_dict, save_directory / "pytorch_model.bin")

        # Save config
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "gliner_config.json")

        # Save tokenizer
        self.data_processor.transformer_tokenizer.save_pretrained(save_directory)

        # Push to hub if requested
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()
            if config is not None:
                kwargs["config"] = config
            if repo_id is None:
                repo_id = save_directory.name
            return self.push_to_hub(repo_id=repo_id, **kwargs)

        return None

    @classmethod
    def _load_config(cls, config_file: Path, **config_overrides) -> object:
        """
        Load configuration from file with optional overrides.

        Args:
            config_file: Path to config file
            **config_overrides: Config parameters to override

        Returns:
            Config instance
        """
        with open(config_file) as f:
            config_dict = json.load(f)

        config_dict.pop("model_type", None)

        # Apply overrides
        for key, value in config_overrides.items():
            if value is not None:
                config_dict[key] = value

        # Use specific config class if defined, otherwise auto-detect
        if cls.config_class is not None:
            config = cls.config_class(**config_dict)
        else:
            config = GLiNERConfig(**config_dict)

        return config

    @staticmethod
    def _set_tokenizer_spec_tokens(tokenizer):
        if hasattr(tokenizer, "add_bos_token"):
            tokenizer.add_bos_token = tokenizer.bos_token_id is not None
        if hasattr(tokenizer, "add_eos_token"):
            tokenizer.add_eos_token = tokenizer.eos_token_id is not None
        return tokenizer

    @classmethod
    def _load_tokenizer(cls, config: GLiNERConfig, model_dir: Path, cache_dir: Optional[Path] = None):
        """
        Load tokenizer from directory.

        Args:
            config: GLiNER config instance
            model_dir: Directory containing tokenizer files
            cache_dir: Cache directory for downloads

        Returns:
            Tokenizer instance or None
        """
        tokenizer_config_path = model_dir / "tokenizer_config.json"

        if tokenizer_config_path.is_file():
            tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)

        return cls._set_tokenizer_spec_tokens(tokenizer)

    @classmethod
    def _load_state_dict(
        cls,
        model_file: Path,
        map_location: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Load state dict from file.

        Args:
            model_file: Path to model file
            map_location: Device to map tensors to
            dtype: If set, floating-point tensors are cast to this dtype during
                loading so the state dict never fully materializes at the
                on-disk precision.

        Returns:
            State dict
        """
        if model_file.suffix == ".safetensors" or str(model_file).endswith(".safetensors"):
            state_dict = {}
            with safe_open(model_file, framework="pt", device=map_location) as f:
                for key in f.keys():  # noqa: SIM118
                    tensor = f.get_tensor(key)
                    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    state_dict[key] = tensor
        else:
            state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
            if dtype is not None:
                for k, v in state_dict.items():
                    if torch.is_tensor(v) and v.is_floating_point() and v.dtype != dtype:
                        state_dict[k] = v.to(dtype)
        return state_dict

    @classmethod
    def _download_model(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        token: Union[str, bool, None] = None,
        local_files_only: bool = False,
        variant: Optional[str] = None,
    ) -> Path:
        """
        Download model from HuggingFace Hub or use local directory.

        Args:
            model_id: Model identifier or local path
            revision: Model revision
            cache_dir: Cache directory
            force_download: Force redownload
            proxies: Proxy configuration
            resume_download: Resume interrupted downloads
            token: HF token
            local_files_only: Only use local files
            variant: If set, restrict the download to ``model.{variant}.safetensors``
                (plus configs/tokenizer assets) via ``snapshot_download``'s
                ``allow_patterns``. Must be canonicalized via
                :meth:`_normalize_variant` by the caller.

        Returns:
            Path to model directory
        """
        model_dir = Path(model_id)

        if not model_dir.exists():
            allow_patterns = cls._variant_allow_patterns(variant) if variant else None
            model_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=allow_patterns,
                )
            )

        return model_dir

    @staticmethod
    def _resize_token_embeddings(instance, config_instance, tokenizer, resize_token_embeddings=True):
        add_tokens = instance._get_special_tokens()
        # Resize token embeddings if needed
        if resize_token_embeddings and (config_instance.class_token_index == -1 or config_instance.vocab_size == -1):
            if tokenizer is not None:
                tokenizer.add_tokens(add_tokens, special_tokens=True)
            instance.resize_embeddings()

    @classmethod
    def load_from_config(
        cls,
        config: Union[str, Path, GLiNERConfig, dict],
        cache_dir: Optional[Union[str, Path]] = None,
        load_tokenizer: bool = True,
        resize_token_embeddings: bool = True,
        backbone_from_pretrained: bool = True,
        compile_torch_model: bool = False,
        quantize: Optional[str] = None,
        map_location: str = "cpu",
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """Initialize a model from configuration without loading pretrained weights.

        This method creates a new model instance from scratch using the provided configuration.
        The backbone encoder can optionally be loaded from pretrained weights, but the GLiNER-specific
        layers are always randomly initialized.

        Args:
            config: Model configuration (GLiNERConfig object, path to config file, or dict).
            cache_dir: Cache directory for downloads.
            load_tokenizer: Whether to load tokenizer.
            resize_token_embeddings: Whether to resize token embeddings.
            backbone_from_pretrained: Whether to load the backbone encoder from pretrained weights.
            compile_torch_model: Whether to compile with torch.compile.
            quantize: Only ``"int8"`` is accepted (int8 dynamic quantization: torchao
                on GPU, FBGEMM on CPU). For precision-only changes (fp16/bf16), use
                ``dtype=``. ``None`` to disable.
            map_location: Device to map model to.
            max_length: Override max_length in config.
            max_width: Override max_width in config.
            post_fusion_schema: Override post_fusion_schema in config.
            _attn_implementation: Override attention implementation.
            **model_kwargs: Additional model initialization arguments.

        Returns:
            Initialized model instance with randomly initialized weights (except backbone if specified).

        Examples:
            >>> config = GLiNERConfig(model_name="microsoft/deberta-v3-small")
            >>> model = GLiNER.load_from_config(config)
            >>> model = GLiNER.load_from_config("path/to/gliner_config.json")
            >>> # Load with pretrained backbone but random GLiNER layers
            >>> model = GLiNER.load_from_config(config, backbone_from_pretrained=True)
        """
        # Load config from various sources
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config_path) as f:
                config_dict = json.load(f)
        elif isinstance(config, dict):
            config_dict = config.copy()
        elif isinstance(config, BaseGLiNERConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(f"config must be a GLiNERConfig object, path to config file, or dict. Got {type(config)}")
        config_dict.pop("model_type", None)
        # Apply config overrides
        if max_length is not None:
            config_dict["max_len"] = max_length
        if max_width is not None:
            config_dict["max_width"] = max_width
        if post_fusion_schema is not None:
            config_dict["post_fusion_schema"] = post_fusion_schema
        if _attn_implementation is not None:
            config_dict["_attn_implementation"] = _attn_implementation

        # Create config instance using the class's config_class
        if cls.config_class is not None:
            config_instance = cls.config_class(**config_dict)
        else:
            config_instance = GLiNERConfig(**config_dict)

        # Load tokenizer if requested
        tokenizer = None
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(config_instance.model_name, cache_dir=cache_dir)
            cls._set_tokenizer_spec_tokens(tokenizer)
        # Create model instance from scratch
        instance = cls(
            config_instance,
            tokenizer=tokenizer,
            backbone_from_pretrained=backbone_from_pretrained,
            cache_dir=cache_dir,
            **model_kwargs,
        )

        cls._resize_token_embeddings(instance, config_instance, tokenizer, resize_token_embeddings)

        # Move to device
        instance.model.to(map_location)

        # Compile if requested
        if compile_torch_model:
            if "cuda" in map_location:
                logger.info("Compiling torch model...")
                instance.compile()
            else:
                warnings.warn(
                    "Cannot compile model on CPU. Set `map_location='cuda'` to compile.",
                    stacklevel=2,
                )

        if quantize:
            dtype = quantize if isinstance(quantize, str) else "fp16"
            instance.quantize(dtype)

        instance.eval()
        return instance

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_dir: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = None,
        resize_token_embeddings: Optional[bool] = True,
        compile_torch_model: Optional[bool] = False,
        quantize: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        variant: Optional[str] = None,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        session_options=None,
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """Load pretrained model from HuggingFace Hub or local directory.

        Args:
            model_id: Model identifier or local path.
            model_dir: Override model directory path.
            revision: Model revision.
            cache_dir: Cache directory.
            force_download: Force redownload.
            proxies: Proxy configuration.
            resume_download: Resume interrupted downloads.
            local_files_only: Only use local files.
            token: HF token for private repos.
            map_location: Device to map model to.
            strict: Enforce strict state_dict loading.
            load_tokenizer: Whether to load tokenizer.
            resize_token_embeddings: Whether to resize embeddings.
            compile_torch_model: Whether to compile with torch.compile.
            quantize: Only ``"int8"`` is accepted (int8 dynamic quantization: torchao
                on GPU, FBGEMM on CPU). For precision-only changes (fp16/bf16), use
                ``dtype=``. ``None`` to disable.
            dtype: Target floating-point dtype for the loaded weights (e.g.
                ``torch.bfloat16``, ``"bf16"``, ``"fp16"``). When set, the model
                shell is pre-cast and each state-dict tensor is cast during
                reading, so the full fp32 copy is never materialized — peak
                host memory is roughly half of the default path for bf16/fp16.
                Prefer this over ``quantize`` for plain precision changes.
            variant: If set (``"fp16"`` or ``"bf16"``), prefer
                ``model.{variant}.safetensors`` over the default fp32 file.
                Best-effort: the loader probes the Hub (or local path) for the
                variant file before downloading. If it is published, only the
                variant file is fetched (~half the bytes vs fp32) and loaded
                directly. If it is not published, a ``UserWarning`` is emitted
                and the loader falls back to the default fp32 file plus an
                in-memory cast — same outcome as ``dtype={variant!r}`` alone,
                no I/O win, no error. ``dtype`` is inferred from ``variant``
                when not set; passing both with mismatched precisions raises.
                ``None`` (default) preserves the prior behavior verbatim.
            load_onnx_model: Whether to load ONNX model instead of PyTorch.
            onnx_model_file: Path to ONNX model file.
            session_options: ONNX runtime session options.
            max_length: Override max_length in config.
            max_width: Override max_width in config.
            post_fusion_schema: Override post_fusion_schema in config.
            _attn_implementation: Override attention implementation.
            **model_kwargs: Additional model initialization arguments.

        Returns:
            Loaded model instance.
        """
        # Resolve variant + dtype up front so the download path can be
        # narrowed *before* hitting the network. Must happen before any
        # snapshot_download call so allow_patterns can apply.
        variant = cls._normalize_variant(variant)
        torch_dtype = cls._parse_dtype(dtype)
        if variant is not None:
            variant_dtype = cls._VARIANT_TO_DTYPE[variant]
            if torch_dtype is None:
                torch_dtype = variant_dtype
            elif torch_dtype != variant_dtype:
                raise ValueError(
                    f"variant={variant!r} requires dtype={variant_dtype}; got dtype={torch_dtype}. "
                    f"Drop dtype= to inherit from variant, or unset variant= to load the default file."
                )
            # Probe Hub/disk for variant availability. If the file isn't
            # published, this warns and returns None — we then fall back to
            # the default fp32 download path with `torch_dtype` already set,
            # so cast-on-read produces the requested precision (no I/O win,
            # but the model still loads). Skip if a model_dir was provided
            # by the caller (they've already resolved the location); the
            # file-resolution step below handles missing files there.
            if model_dir is None:
                variant = cls._resolve_variant(
                    model_id,
                    variant,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )

        # Download or locate model
        if model_dir is None:
            model_dir = cls._download_model(
                model_id,
                revision,
                cache_dir,
                force_download,
                proxies,
                resume_download,
                token,
                local_files_only,
                variant=variant,
            )

        # Load config
        config_file = model_dir / "gliner_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config file found in {model_dir}")

        config = cls._load_config(
            config_file,
            max_len=max_length,
            max_width=max_width,
            post_fusion_schema=post_fusion_schema,
            _attn_implementation=_attn_implementation,
        )

        # Load tokenizer
        if load_tokenizer is None:
            load_tokenizer = True

        tokenizer = None
        if load_tokenizer:
            tokenizer = cls._load_tokenizer(config, model_dir, cache_dir)

        if not load_onnx_model:
            # Find the model file. _resolve_model_file picks the variant file
            # if present, falls back to the default fp32 file with a warning if
            # the variant is missing (e.g. caller passed model_dir directly to
            # a local path without a variant file). torch_dtype is already set
            # from the variant earlier, so the cast-on-read in _load_state_dict
            # still produces the requested precision after a fallback.
            model_file, _ = cls._resolve_model_file(model_dir, variant)

            # Create model instance
            instance = cls(
                config, tokenizer=tokenizer, backbone_from_pretrained=False, cache_dir=cache_dir, **model_kwargs
            )

            cls._resize_token_embeddings(instance, config, tokenizer, resize_token_embeddings)

            if torch_dtype is not None:
                # Pre-cast the random-init shell so the model never exists at
                # fp32 alongside the loaded state dict. ``.to(floating_dtype)``
                # only touches floating-point params/buffers.
                instance.model.to(torch_dtype)

            # Load state dict (tensors cast to ``torch_dtype`` during read when set)
            state_dict = cls._load_state_dict(model_file, map_location, dtype=torch_dtype)
            instance.model.load_state_dict(state_dict, strict=strict)
            del state_dict
            instance.model.to(map_location)

            if compile_torch_model:
                if "cuda" in map_location:
                    logger.info("Compiling torch model...")
                    instance.compile()
                else:
                    warnings.warn("Cannot compile model on CPU. Set `map_location='cuda'` to compile.", stacklevel=2)

            if quantize:
                if quantize is True:
                    raise ValueError(
                        "`quantize=True` is no longer supported. Use `quantize='int8'` for "
                        "int8 quantization, or `dtype='fp16'`/`'bf16'` on `from_pretrained` "
                        "for precision-only changes."
                    )
                instance.quantize(quantize)

            instance.eval()
        else:
            model_file = model_dir / onnx_model_file
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"The ONNX model can't be loaded from {model_file}.")
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            if "cuda" in map_location:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available but `map_location` is set to 'cuda'.")
                providers = ["CUDAExecutionProvider"]
            ort_session = ort.InferenceSession(model_file, session_options, providers=providers)
            model = cls.ort_model_class(ort_session)
            instance = cls(config, tokenizer=tokenizer, model=model)

        return instance

    def _check_onnx_export_preconditions(self):
        if self.onnx_model:
            raise RuntimeError(
                "This instance already wraps an ONNX/ORT model. Export is intended for PyTorch-based models."
            )
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not available. Install `onnxruntime` to export to ONNX.")
        if not hasattr(self, "data_processor") or not hasattr(self, "data_collator_class"):
            raise RuntimeError("Model is not fully initialized (missing data_processor or data_collator).")

    def _build_dummy_batch(
        self,
        labels: Optional[list[str]] = None,
        text: str = "ONNX export dummy input.",
    ) -> dict[str, torch.Tensor]:
        """
        Build a single CPU batch using the model's own preprocessing stack.

        Concrete exporters can call this and then select the keys they need.
        """
        if labels is None or len(labels) == 0:
            labels = ["person", "organization", "country"]

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokens, _, _ = self.prepare_inputs(texts)
        input_x = self.prepare_base_input(tokens)

        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=False,
            return_entities=False,
            return_id_to_classes=False,
            prepare_labels=False,
        )

        def collate_fn(batch, entity_types=labels):
            try:
                return collator(batch, entity_types=entity_types)
            except TypeError:
                return collator(batch)

        loader = DataLoader(input_x, batch_size=1, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(loader))

        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cpu")

        return batch

    def _run_torch_onnx_export(
        self,
        wrapper: nn.Module,
        all_inputs: tuple,
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: dict[str, dict[int, str]],
        onnx_path: Path,
        opset: int,
    ):
        wrapper.eval()
        torch.onnx.export(
            wrapper,
            all_inputs,
            f=str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )

    def _maybe_quantize_onnx(
        self,
        onnx_path: Path,
        quantized_path: Path,
        quantize: bool,
    ) -> Optional[Path]:
        if not quantize:
            return None

        if quantize_dynamic is None:
            warnings.warn("onnxruntime.quantization is not available; skipping quantization.", stacklevel=2)
            return None

        try:
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QUInt8,
            )
            return quantized_path
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}", stacklevel=2)
            return None

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Define ONNX input specification for this model type.

        Must be implemented by child classes that support ONNX export.

        Returns:
            Dictionary with:
                - input_names: List of input tensor names
                - output_names: List of output tensor names
                - dynamic_axes: Dynamic axis specifications
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_onnx_input_spec() or override export_to_onnx() entirely."
        )

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """
        Create ONNX export wrapper.

        Default implementation creates a simple passthrough wrapper.
        Override this method for custom wrapper logic.

        Args:
            core_model: The model to wrap

        Returns:
            Wrapped model for ONNX export
        """

        class DefaultWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, *args, **kwargs):
                out = self.core(*args, **kwargs)
                return out.logits if hasattr(out, "logits") else out[0]

        return DefaultWrapper(core_model)

    def _get_onnx_export_kwargs(self) -> dict[str, Any]:
        """Get additional kwargs for ONNX export (e.g., labels for bi-encoders).

        Override in child classes as needed.

        Returns:
            Dictionary of kwargs to pass to _build_dummy_batch
        """
        return {}

    def _prepare_onnx_batch(self, batch: dict[str, torch.Tensor], **export_kwargs) -> tuple[tuple, dict[str, Any]]:
        """
        Prepare batch for ONNX export. Can be overridden for special preprocessing.

        Args:
            batch: Dummy batch from _build_dummy_batch
            **export_kwargs: Additional export arguments

        Returns:
            Tuple of (input_tuple, updated_spec) where input_tuple contains the actual
            inputs to pass to the wrapper and updated_spec may have modified input_names/dynamic_axes
        """
        spec = self._get_onnx_input_spec()
        all_inputs = tuple(batch[name] for name in spec["input_names"])
        return all_inputs, spec

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
        **export_kwargs,
    ) -> dict[str, Optional[str]]:
        """Unified ONNX export method using specifications from child classes.

        Args:
            save_dir: Directory to save ONNX files.
            onnx_filename: Name of the ONNX model file.
            quantized_filename: Name of the quantized model file.
            quantize: Whether to create a quantized version.
            opset: ONNX opset version.
            **export_kwargs: Additional export arguments (model-specific).

        Returns:
            Dictionary with paths to exported models:
                - onnx_path: Path to standard ONNX model
                - quantized_path: Path to quantized model (if quantize=True)
        """
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        # Merge export kwargs with model-specific kwargs
        batch_kwargs = {**self._get_onnx_export_kwargs(), **export_kwargs}
        batch = self._build_dummy_batch(**batch_kwargs)

        core = self.model.to("cpu").eval()

        # Prepare inputs and get spec (allows for dynamic modification)
        all_inputs, spec = self._prepare_onnx_batch(batch, **export_kwargs)

        # Create wrapper
        wrapper = self._create_onnx_wrapper(core)

        # Export
        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            spec["input_names"],
            spec["output_names"],
            spec["dynamic_axes"],
            onnx_path,
            opset,
        )

        # Save Config file
        self.config.to_json_file(save_dir / "gliner_config.json")
        # Save Tokenizer file
        self.data_processor.transformer_tokenizer.save_pretrained(save_dir)

        # Quantize if requested
        q_path = self._maybe_quantize_onnx(onnx_path, save_dir / quantized_filename, quantize)

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }

    def _create_data_collator(self, **kwargs):
        """
        Create data collator. Override in child classes if needed.

        Returns:
            Data collator instance
        """
        return self.data_collator_class(self.config, data_processor=self.data_processor, prepare_labels=True, **kwargs)

    def _get_freezable_components(self):
        """
        Get dictionary mapping component names to their actual modules.

        Returns:
            dict: Mapping of component names to module objects
        """
        components = {}

        # Text encoder (always present)
        if (
            hasattr(self, "model")
            and hasattr(self.model, "token_rep_layer")
            and hasattr(self.model.token_rep_layer, "bert_layer")
        ):
            components["text_encoder"] = self.model.token_rep_layer.bert_layer.model

        # Labels encoder (optional)
        if (
            self.config.labels_encoder is not None
            and hasattr(self.model, "token_rep_layer")
            and hasattr(self.model.token_rep_layer, "labels_encoder")
        ):
            components["labels_encoder"] = self.model.token_rep_layer.labels_encoder.model

        # Decoder (optional)
        if (
            self.config.labels_decoder is not None
            and hasattr(self.model, "decoder")
            and hasattr(self.model.decoder, "decoder_layer")
        ):
            components["decoder"] = self.model.decoder.decoder_layer.model

        return components

    def freeze_component(self, component_name: str):
        """
        Freeze a specific component of the model.

        Args:
            component_name: Name of component to freeze (e.g., 'text_encoder', 'labels_encoder', 'decoder')
        """
        components = self._get_freezable_components()
        if component_name in components:
            components[component_name].requires_grad_(False)
            logger.info("Frozen: %s", component_name)
        else:
            available = ", ".join(components.keys())
            warnings.warn(f"Component '{component_name}' not found. Available components: {available}", stacklevel=2)

    def unfreeze_component(self, component_name: str):
        """
        Unfreeze a specific component of the model.

        Args:
            component_name: Name of component to unfreeze
        """
        components = self._get_freezable_components()
        if component_name in components:
            components[component_name].requires_grad_(True)
            logger.info("Unfrozen: %s", component_name)
        else:
            available = ", ".join(components.keys())
            warnings.warn(f"Component '{component_name}' not found. Available components: {available}", stacklevel=2)

    @classmethod
    def create_training_args(
        cls,
        output_dir: Union[str, Path],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        others_lr: Optional[float] = None,
        others_weight_decay: Optional[float] = None,
        focal_loss_alpha: float = -1,
        focal_loss_gamma: float = 0.0,
        rel_focal_loss_alpha: Optional[float] = None,
        rel_focal_loss_gamma: Optional[float] = None,
        focal_loss_prob_margin: float = 0.0,
        loss_reduction: str = "sum",
        negatives: float = 1.0,
        masking: str = "none",
        lr_scheduler_type: str = "linear",
        warmup_ratio: float = 0.1,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        max_grad_norm: float = 1.0,
        max_steps: int = 10000,
        save_steps: int = 1000,
        save_total_limit: int = 10,
        logging_steps: int = 10,
        use_cpu: bool = False,
        bf16: bool = False,
        dataloader_num_workers: int = 1,
        report_to: str = "none",
        **kwargs,
    ) -> TrainingArguments:
        """Create training arguments with sensible defaults.

        Args:
            output_dir: Directory to save model checkpoints.
            learning_rate: Learning rate for main parameters.
            weight_decay: Weight decay for main parameters.
            others_lr: Learning rate for other parameters.
            others_weight_decay: Weight decay for other parameters.
            focal_loss_alpha: Alpha for focal loss.
            focal_loss_gamma: Gamma for focal loss.
            rel_focal_loss_alpha: Alpha for relation focal loss. Defaults to entity alpha.
            rel_focal_loss_gamma: Gamma for relation focal loss. Defaults to entity gamma.
            focal_loss_prob_margin: Probability margin for focal loss.
            loss_reduction: Loss reduction method.
            negatives: Negative sampling ratio.
            masking: Masking strategy.
            lr_scheduler_type: Learning rate scheduler type.
            warmup_ratio: Warmup ratio.
            per_device_train_batch_size: Training batch size.
            per_device_eval_batch_size: Evaluation batch size.
            max_grad_norm: Maximum gradient norm.
            max_steps: Maximum training steps.
            save_steps: Save checkpoint every N steps.
            save_total_limit: Maximum number of checkpoints to keep.
            logging_steps: Log every N steps.
            use_cpu: Whether to use CPU.
            bf16: Whether to use bfloat16.
            dataloader_num_workers: Number of dataloader workers.
            report_to: Where to report metrics.
            **kwargs: Additional training arguments.

        Returns:
            TrainingArguments instance.
        """
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            others_lr=others_lr or learning_rate,
            others_weight_decay=others_weight_decay or weight_decay,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            rel_focal_loss_alpha=rel_focal_loss_alpha,
            rel_focal_loss_gamma=rel_focal_loss_gamma,
            focal_loss_prob_margin=focal_loss_prob_margin,
            loss_reduction=loss_reduction,
            negatives=negatives,
            masking=masking,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            dataloader_num_workers=dataloader_num_workers,
            logging_steps=logging_steps,
            use_cpu=use_cpu,
            report_to=report_to,
            bf16=bf16,
            **kwargs,
        )

    def train_model(
        self,
        train_dataset,
        eval_dataset,
        training_args: Optional[TrainingArguments] = None,
        freeze_components: Optional[list[str]] = None,
        compile_model: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        **training_kwargs,
    ) -> Trainer:
        """Train the model.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            training_args: Training arguments (created with defaults if None).
            freeze_components: List of component names to freeze (e.g., ['text_encoder', 'decoder']).
            compile_model: Whether to compile model with torch.compile.
            output_dir: Output directory (required if training_args is None).
            **training_kwargs: Additional kwargs for creating training args.

        Returns:
            Trained Trainer instance.
        """
        # Create training arguments if not provided
        if training_args is None:
            if output_dir is None:
                raise ValueError("Either training_args or output_dir must be provided")
            training_args = self.create_training_args(output_dir=output_dir, **training_kwargs)

        # Compile model if requested
        if compile_model:
            self.compile()

        # Freeze components if specified
        if freeze_components:
            for component_name in freeze_components:
                self.freeze_component(component_name)

        # Create data collator
        data_collator = self._create_data_collator()

        # Create trainer with version-conditional tokenizer argument
        # transformers < 5.0 requires tokenizer, >= 5.0 does not
        trainer_kwargs = {
            "model": self,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
        }

        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            trainer_kwargs["tokenizer"] = self.data_processor.transformer_tokenizer
        else:
            trainer_kwargs["processing_class"] = self.data_processor.transformer_tokenizer

        trainer = Trainer(**trainer_kwargs)

        # Train
        trainer.train()

        return trainer


class BaseEncoderGLiNER(BaseGLiNER):
    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
        self.model = self.model_class(config, backbone_from_pretrained, cache_dir=cache_dir, **kwargs)
        return self.model

    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)
        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter)
        return self.data_processor

    def set_class_indices(self):
        """Set the class token index in the configuration based on tokenizer vocabulary."""
        self.config.class_token_index = len(self.data_processor.transformer_tokenizer) - 2

    def resize_embeddings(self, set_class_token_index=True):
        """Resize token embeddings to match tokenizer vocabulary size.

        Args:
            set_class_token_index: Whether to update the class token index.
        """
        if set_class_token_index:
            self.set_class_indices()

        if len(self.data_processor.transformer_tokenizer) != self.config.vocab_size:
            new_num_tokens = len(self.data_processor.transformer_tokenizer)
            model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, None)
            self.config.vocab_size = model_embeds.num_embeddings
            if hasattr(self.config, "encoder_config"):
                self.config.encoder_config.vocab_size = model_embeds.num_embeddings

    def prepare_inputs(self, texts: List[str]):
        """Prepare inputs for the model by tokenizing and creating index mappings.

        Args:
            texts: The input texts to process.

        Returns:
            Tuple containing:
                - all_tokens: List of tokenized texts
                - all_start_token_idx_to_text_idx: Start position mappings
                - all_end_token_idx_to_text_idx: End position mappings
        """
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.data_processor.words_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)
        return all_tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx

    def prepare_base_input(self, all_tokens: List[List[str]]) -> List[Dict[str, Any]]:
        """Prepare base input format for data collation.

        Args:
            all_tokens: List of tokenized texts.

        Returns:
            List of input dictionaries ready for collation.
        """
        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        return input_x

    def _filter_valid_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Filter out empty or whitespace-only strings from input texts.

        Args:
            texts: List of input texts.

        Returns:
            Tuple containing:
                - valid_texts: List of non-empty texts
                - valid_to_orig_idx: Mapping from valid text index to original text index
        """
        valid_texts = []
        valid_to_orig_idx = []

        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_to_orig_idx.append(i)

        return valid_texts, valid_to_orig_idx

    def _convert_spans_to_word_indices(
        self,
        input_spans: List[List[Dict]],
        all_start_token_idx_to_text_idx: List[List[int]],
        all_end_token_idx_to_text_idx: List[List[int]],
    ) -> List[List[Tuple[int, int]]]:
        """Convert character-level input spans to word-level (start, end) tuples.

        Args:
            input_spans: Per-text list of span dicts with 'start' and 'end' char positions.
            all_start_token_idx_to_text_idx: Per-text mapping from word index to char start.
            all_end_token_idx_to_text_idx: Per-text mapping from word index to char end.

        Returns:
            Per-text list of (word_start, word_end) tuples. Spans that don't align
            to word boundaries are silently dropped.
        """
        word_input_spans = []
        for text_i, spans in enumerate(input_spans):
            # Build reverse lookups: char position -> word index
            start_char_to_word = {
                char_pos: word_idx for word_idx, char_pos in enumerate(all_start_token_idx_to_text_idx[text_i])
            }
            end_char_to_word = {
                char_pos: word_idx for word_idx, char_pos in enumerate(all_end_token_idx_to_text_idx[text_i])
            }

            word_spans = []
            for span in spans:
                word_start = start_char_to_word.get(span["start"])
                word_end = end_char_to_word.get(span["end"])
                if word_start is not None and word_end is not None and word_end >= word_start:
                    word_spans.append((word_start, word_end))
            word_input_spans.append(word_spans)
        return word_input_spans

    def _map_entities_to_original(
        self,
        outputs: List[List[Any]],
        valid_to_orig_idx: List[int],
        all_start_token_idx_to_text_idx: List[List[int]],
        all_end_token_idx_to_text_idx: List[List[int]],
        valid_texts: List[str],
        num_original_texts: int,
    ) -> List[List[Dict[str, Any]]]:
        """Map entity predictions back to original text indices.

        Args:
            outputs: Decoded outputs from model.
            valid_to_orig_idx: Mapping from valid index to original index.
            all_start_token_idx_to_text_idx: Start position mappings.
            all_end_token_idx_to_text_idx: End position mappings.
            valid_texts: Valid (non-empty) texts.
            num_original_texts: Total number of original texts.

        Returns:
            List of entity predictions aligned with original input.
        """
        all_entities = [[] for _ in range(num_original_texts)]

        for valid_i, output in enumerate(outputs):
            orig_i = valid_to_orig_idx[valid_i]
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[valid_i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[valid_i]

            entities = []
            for span in output:
                # Use Span object attributes
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]

                entity = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": valid_texts[valid_i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }

                if span.class_probs is not None:
                    entity["class_probs"] = span.class_probs

                entities.append(entity)

            all_entities[orig_i] = entities

        return all_entities

    def _process_batches(
        self,
        data_loader,
        threshold,
        flat_ner,
        multi_label,
        packing_config=None,
        return_class_probs=False,
        word_input_spans=None,
        **external_inputs,
    ):
        """Shared batch processing logic using modular run_batch and decode_batch."""
        outputs = []
        batch_offset = 0

        for batch in data_loader:
            model_output = self.run_batch(
                batch,
                threshold=threshold,
                packing_config=packing_config,
                move_to_device=True,
                **external_inputs,
            )

            batch_input_spans = None
            if word_input_spans is not None:
                current_batch_size = len(batch["tokens"])
                batch_input_spans = word_input_spans[batch_offset : batch_offset + current_batch_size]
                batch_offset += current_batch_size

            decoded = self.decode_batch(
                model_output,
                batch,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                input_spans=batch_input_spans,
            )
            outputs.extend(decoded)

        return outputs

    def prepare_batch(
        self,
        texts: Union[str, List[str]],
        labels: Union[str, List[str], List[List[str]]],
        input_spans: Optional[List[List[Dict]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare raw inputs for inference (tokenization and normalization).

        This method handles text normalization, tokenization, and span conversion.
        Use this as the first step in the inference pipeline.

        Args:
            texts: Single text string or list of texts.
            labels: Entity labels - string, list of strings, or per-text label lists.
            input_spans: Optional pre-defined spans to classify (character positions).
            **kwargs: Additional keyword arguments passed to the data processor.

        Returns:
            Dictionary containing:
                - input_x: List of input dicts ready for collation
                - tokens: Tokenized texts
                - start_token_map: Per-text mapping from token idx to char start
                - end_token_map: Per-text mapping from token idx to char end
                - word_input_spans: Spans converted to word indices (or None)
                - entity_types: Normalized entity types
                - valid_texts: Non-empty texts that will be processed
                - valid_to_orig_idx: Mapping from valid indices to original indices
                - num_original: Total number of original texts
        """
        if isinstance(texts, str):
            texts = [texts]

        num_original = len(texts)
        valid_texts, valid_to_orig_idx = self._filter_valid_texts(texts)

        if not valid_texts:
            return {
                "input_x": [],
                "tokens": [],
                "start_token_map": [],
                "end_token_map": [],
                "word_input_spans": None,
                "entity_types": [],
                "valid_texts": [],
                "valid_to_orig_idx": [],
                "num_original": num_original,
            }

        if isinstance(labels, str):
            entity_types = list(dict.fromkeys([labels]))
        elif labels and isinstance(labels[0], list):
            entity_types = [list(dict.fromkeys(lbls)) for lbls in labels]
        else:
            entity_types = list(dict.fromkeys(labels))

        tokens, start_token_map, end_token_map = self.prepare_inputs(valid_texts)

        word_input_spans = None
        if input_spans is not None:
            valid_input_spans = [input_spans[i] for i in valid_to_orig_idx]
            word_input_spans = self._convert_spans_to_word_indices(valid_input_spans, start_token_map, end_token_map)

        input_x = self.prepare_base_input(tokens)

        return {
            "input_x": input_x,
            "tokens": tokens,
            "start_token_map": start_token_map,
            "end_token_map": end_token_map,
            "word_input_spans": word_input_spans,
            "entity_types": entity_types,
            "valid_texts": valid_texts,
            "valid_to_orig_idx": valid_to_orig_idx,
            "num_original": num_original,
        }

    def collate_batch(
        self,
        input_x: List[Dict[str, Any]],
        entity_types: Union[List[str], List[List[str]]],
        collator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Collate prepared inputs into a tensor batch.

        Args:
            input_x: List of input dicts from prepare_batch.
            entity_types: Entity type labels.
            collator: Optional pre-created collator instance. If None, creates one.

        Returns:
            Collated batch dictionary with tensors ready for the model.
        """
        if collator is None:
            collator = self.data_collator_class(
                self.config,
                data_processor=self.data_processor,
                return_tokens=True,
                return_entities=True,
                return_id_to_classes=True,
                prepare_labels=False,
            )

        batch = collator(input_x, entity_types=entity_types)
        return batch

    @torch.inference_mode()
    def run_batch(
        self,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        packing_config: Optional[InferencePackingConfig] = None,
        move_to_device: bool = True,
        **external_inputs,
    ) -> Any:
        """Run model forward pass on a collated batch.

        Args:
            batch: Collated batch from collate_batch.
            threshold: Confidence threshold for predictions.
            packing_config: Optional inference packing configuration.
            move_to_device: Whether to move tensors to model device.
            **external_inputs: Additional inputs to pass to the model.

        Returns:
            Model output containing logits and span information.
        """
        if move_to_device and not self.onnx_model:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if packing_config is not None or external_inputs:
            model_inputs = {**batch, **external_inputs}
            if packing_config is not None:
                model_inputs["packing_config"] = packing_config
        else:
            model_inputs = batch

        model_output = self.model(**model_inputs, threshold=threshold)
        return model_output

    def decode_batch(
        self,
        model_output: Any,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        input_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> List[List[Any]]:
        """Decode model output into entity predictions.

        Args:
            model_output: Output from run_batch.
            batch: The collated batch (needs 'tokens' and 'id_to_classes').
            threshold: Confidence threshold for predictions.
            flat_ner: Whether to use flat NER (no overlapping entities).
            multi_label: Whether to allow multiple labels per span.
            return_class_probs: Whether to include class probabilities.
            input_spans: Optional word-level input spans to classify.

        Returns:
            List of entity lists (one per text in batch).
        """
        model_logits = model_output[0]
        if not isinstance(model_logits, torch.Tensor):
            model_logits = torch.from_numpy(model_logits)

        decoded = self.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            model_logits,
            span_idx=model_output.span_idx,
            span_mask=model_output.span_mask,
            span_logits=model_output.span_logits,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            input_spans=input_spans,
        )
        return decoded

    def map_entities_to_text(
        self,
        decoded: List[List[Any]],
        valid_texts: List[str],
        valid_to_orig_idx: List[int],
        start_token_map: List[List[int]],
        end_token_map: List[List[int]],
        num_original: int,
    ) -> List[List[Dict[str, Any]]]:
        """Map decoded entities back to character positions in original texts.

        Args:
            decoded: Decoded entity spans from decode_batch.
            valid_texts: List of valid (non-empty) texts.
            valid_to_orig_idx: Mapping from valid indices to original indices.
            start_token_map: Per-text token-to-char-start mapping.
            end_token_map: Per-text token-to-char-end mapping.
            num_original: Total number of original texts.

        Returns:
            List of entity dicts aligned with original input texts.
        """
        return self._map_entities_to_original(
            decoded,
            valid_to_orig_idx,
            start_token_map,
            end_token_map,
            valid_texts,
            num_original,
        )

    def create_collator(self) -> Any:
        """Create a data collator instance for batch collation.

        Useful for serve.py to create a reusable collator.

        Returns:
            Configured data collator instance.
        """
        return self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

    @torch.no_grad()
    def inference(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        packing_config: Optional[InferencePackingConfig] = None,
        input_spans: Optional[List[List[Dict]]] = None,
        return_class_probs: bool = False,
        **external_inputs,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities for a batch of texts.

        Args:
            texts: A list of input texts to predict entities for or a single text string.
            labels: A list of labels to predict.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            multi_label: Whether to allow multiple labels per token. Defaults to False.
            batch_size: Batch size for processing. Defaults to 8.
            packing_config: Configuration describing how to pack encoder inputs. When None
                the instance-level configuration set via configure_inference_packing is used.
            input_spans: Input entity spans that should be classified by the model.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **external_inputs: Additional inputs to pass to the model.

        Returns:
            List of lists with predicted entities, where each entity is a dictionary containing:
                - start: Start character position
                - end: End character position
                - text: Entity text
                - label: Entity type
                - score: Confidence score
                - class_probs: (optional) Dictionary mapping class names to probabilities (top 5)
        """
        self.eval()

        prepared = self.prepare_batch(texts, labels, input_spans)

        if not prepared["valid_texts"]:
            return [[] for _ in range(prepared["num_original"])]

        collator = self.create_collator()

        def collate_fn(batch):
            return self.collate_batch(batch, prepared["entity_types"], collator)

        data_loader = torch.utils.data.DataLoader(
            prepared["input_x"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        active_packing = packing_config if packing_config is not None else self._inference_packing_config

        outputs = self._process_batches(
            data_loader,
            threshold,
            flat_ner,
            multi_label,
            packing_config=active_packing,
            return_class_probs=return_class_probs,
            word_input_spans=prepared["word_input_spans"],
            **external_inputs,
        )

        all_entities = self.map_entities_to_text(
            outputs,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        return all_entities

    def predict_entities(
        self,
        text: str,
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        return_class_probs: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            multi_label: Whether to allow multiple labels per entity. Defaults to False.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **kwargs: Additional arguments passed to inference.

        Returns:
            List of entity predictions as dictionaries.
        """
        return self.inference(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            **kwargs,
        )[0]

    def batch_predict_entities(
        self,
        texts: List[str],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities for multiple texts.

        DEPRECATED: Use `inference` instead.

        This method will be removed in a future release. It now forwards to
        `GLiNER.inference(...)` to perform inference.

        Args:
            texts: Input texts.
            labels: Labels to predict.
            flat_ner: Use flat NER. Defaults to True.
            threshold: Confidence threshold. Defaults to 0.5.
            multi_label: Allow multiple labels per token/entity. Defaults to False.
            **kwargs: Extra arguments forwarded to inference (e.g., batch_size).

        Returns:
            List of entity predictions for each text.
        """
        warnings.warn(
            "GLiNER.batch_predict_entities is deprecated and will be removed in a future release. "
            "Please use GLiNER.inference instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.inference(
            texts,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs,
        )

    @torch.no_grad()
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        flat_ner: bool = False,
        multi_label: bool = False,
        threshold: float = 0.5,
        batch_size: int = 12,
    ) -> Tuple[Any, float]:
        """Evaluate the model on a given test dataset.

        Args:
            test_data: The test data containing text and entity annotations.
            flat_ner: Whether to use flat NER. Defaults to False.
            multi_label: Whether to use multi-label classification. Defaults to False.
            threshold: The threshold for predictions. Defaults to 0.5.
            batch_size: The batch size for evaluation. Defaults to 12.

        Returns:
            Tuple containing the evaluation output and the F1 score.
        """
        self.eval()
        # Create the dataset and data loader
        dataset = test_data
        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        all_preds = self._process_batches(data_loader, threshold, flat_ner, multi_label)
        all_trues = []

        # Iterate over data batches
        for batch in data_loader:
            all_trues.extend(batch["entities"])

        # Evaluate the predictions
        evaluator = BaseNEREvaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()

        return out, f1

    def compress_prompt_embeddings(
        self,
        texts: List[str],
        labels: List[str],
        rel_labels: Optional[List[str]] = None,
        batch_size: int = 8,
        distill: bool = False,
        distill_threshold: float = 0.3,
        distill_epochs: int = 3,
        distill_lr: float = 1e-5,
        distill_batch_size: Optional[int] = None,
        distill_output_dir: str = "./distill_ckpt",
        distill_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Precompute averaged prompt embeddings for each label.

        Runs the normal forward pass over (texts, labels) pairs, extracts the
        per-label prompt embedding from each example, and stores the mean per
        label on the underlying model. Sets ``config.precomputed_prompts_mode``
        to True so subsequent inference/training will skip label-prepending and
        look up the stored embeddings instead. Relation labels are supported for
        relation-extraction models via ``rel_labels``.

        When ``distill=True``, the raw (pre-compression) model first generates
        pseudo-labels over ``texts``; the method then compresses prompt
        embeddings and fine-tunes the compressed model on those pseudo-labels
        so quality recovers end-to-end in a single call.

        Args:
            texts: List of raw input texts used as contexts for averaging.
            labels: Entity labels to compress.
            rel_labels: Optional relation labels (relex models only).
            batch_size: Batch size used while running the model.
            distill: If True, generate pseudo-labels with the raw model over
                ``texts`` and fine-tune the compressed model on them.
            distill_threshold: Confidence threshold for pseudo-label generation.
            distill_epochs: Number of fine-tuning epochs.
            distill_lr: Fine-tuning learning rate.
            distill_batch_size: Batch size for fine-tuning (defaults to ``batch_size``).
            distill_output_dir: Output directory passed to ``train_model``.
            distill_train_kwargs: Extra kwargs forwarded to ``train_model``.
        """
        if not texts or not labels:
            raise ValueError("`texts` and `labels` must both be non-empty.")

        distill_data = None
        if distill:
            self.eval()
            with torch.no_grad():
                preds = self.inference(
                    texts,
                    labels,
                    flat_ner=True,
                    threshold=distill_threshold,
                    batch_size=batch_size,
                )
            distill_data = [self._predictions_to_word_level(t, p) for t, p in zip(texts, preds)]
            for sample in distill_data:
                sample["ner_labels"] = labels

        self._compute_prompt_embeddings(
            texts=texts,
            labels=labels,
            rel_labels=rel_labels,
            batch_size=batch_size,
        )

        if distill_data:
            train_kwargs = {
                "num_train_epochs": distill_epochs,
                "max_steps": -1,
                "per_device_train_batch_size": distill_batch_size or batch_size,
                "learning_rate": distill_lr,
                "save_strategy": "no",
                "report_to": "none",
                "logging_steps": 10,
                "remove_unused_columns": False,
            }
            if distill_train_kwargs:
                train_kwargs.update(distill_train_kwargs)
            self.train_model(
                train_dataset=distill_data,
                eval_dataset=None,
                output_dir=distill_output_dir,
                **train_kwargs,
            )
            self.eval()

    @staticmethod
    def _predictions_to_word_level(text: str, preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert char-offset predictions to whitespace-word-level NER format."""
        words = text.split()
        char_starts, char_ends = [], []
        cursor = 0
        remaining = text
        for w in words:
            idx = remaining.find(w)
            abs_start = cursor + idx
            char_starts.append(abs_start)
            char_ends.append(abs_start + len(w))
            cursor = abs_start + len(w)
            remaining = text[cursor:]
        start_to_widx = {s: i for i, s in enumerate(char_starts)}
        end_to_widx = {e: i for i, e in enumerate(char_ends)}
        ner = []
        for p in preds:
            s, e, cls = p["start"], p["end"], p["label"].lower()
            span_text = text[s:e]
            ls = len(span_text) - len(span_text.lstrip())
            le = len(span_text) - len(span_text.rstrip())
            s2, e2 = s + ls, e - le
            if s2 in start_to_widx and e2 in end_to_widx:
                ner.append((start_to_widx[s2], end_to_widx[e2], cls))
        return {"tokenized_text": words, "ner": ner}

    @torch.no_grad()
    def _compute_prompt_embeddings(
        self,
        texts: List[str],
        labels: List[str],
        rel_labels: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> None:
        self.eval()
        device = self.device
        D = self.config.hidden_size

        # Force the normal (non-precomputed) code path while we compute the stats.
        prev_mode = getattr(self.config, "precomputed_prompts_mode", None)
        self.config.precomputed_prompts_mode = False
        try:
            labels = list(dict.fromkeys(labels))
            rel_labels = list(dict.fromkeys(rel_labels)) if rel_labels else None

            L = len(labels)
            L_rel = len(rel_labels) if rel_labels else 0
            ent_sum = torch.zeros(L, D, device=device)
            rel_sum = torch.zeros(L_rel, D, device=device) if L_rel else None

            tokens, _, _ = self.prepare_inputs(texts)
            input_x = self.prepare_base_input(tokens)

            collator_kwargs = {
                "return_tokens": True,
                "return_entities": True,
                "return_id_to_classes": True,
                "prepare_labels": False,
            }
            if rel_labels is not None:
                collator_kwargs["return_rel_id_to_classes"] = True

            collator = self.data_collator_class(self.config, data_processor=self.data_processor, **collator_kwargs)

            def collate_fn(batch):
                if rel_labels is not None:
                    return collator(batch, entity_types=labels, relation_types=rel_labels)
                return collator(batch, entity_types=labels)

            loader = DataLoader(input_x, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            for batch in tqdm(loader, desc="Compressing prompts"):
                batch_gpu = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                input_ids = batch_gpu["input_ids"]
                attention_mask = batch_gpu["attention_mask"]

                # Encode once; pull pre-projection prompt embeddings directly.
                token_embeds = self.model.token_rep_layer(input_ids, attention_mask)
                batch_size_e, _, embed_dim_e = token_embeds.shape

                prompts_emb, _ = extract_prompt_features(
                    self.config.class_token_index,
                    token_embeds,
                    input_ids,
                    attention_mask,
                    batch_size_e,
                    embed_dim_e,
                    self.config.embed_ent_token,
                )
                # prompts_emb is (B, L, D) in the canonical `labels` order since we
                ent_sum = ent_sum + prompts_emb.detach().sum(dim=0).to(device)

                if rel_labels is not None and getattr(self.config, "rel_token_index", None) is not None:
                    rel_emb, _ = extract_prompt_features(
                        self.config.rel_token_index,
                        token_embeds,
                        input_ids,
                        attention_mask,
                        batch_size_e,
                        embed_dim_e,
                        self.config.embed_rel_token,
                    )
                    rel_sum = rel_sum + rel_emb.detach().sum(dim=0).to(device)

            avg = ent_sum / len(texts)
            self.model.set_precomputed_prompts(labels, avg, rel=False)
            self.config.id_to_classes = {i + 1: label for i, label in enumerate(labels)}

            if rel_labels is not None:
                rel_avg = rel_sum / len(texts)
                self.model.set_precomputed_prompts(rel_labels, rel_avg, rel=True)
                self.config.rel_id_to_classes = {i + 1: label for i, label in enumerate(rel_labels)}

        except Exception:
            self.config.precomputed_prompts_mode = prev_mode
            raise

        self.config.precomputed_prompts_mode = True


class BaseBiEncoderGLiNER(BaseEncoderGLiNER):
    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)

        self.data_processor = self.data_processor_class(
            config, tokenizer, words_splitter, labels_tokenizer=labels_tokenizer
        )
        return self.data_processor

    def resize_embeddings(self, **kwargs):
        warnings.warn("Resizing embeddings is not supported for bi-encoder models.", stacklevel=2)

    @torch.no_grad()
    def encode_labels(self, labels: List[str], batch_size: int = 8) -> torch.FloatTensor:
        """Compute embeddings for labels using the label encoder.

        Args:
            labels: A list of labels to encode.
            batch_size: Batch size for processing labels.

        Returns:
            Tensor containing label embeddings with shape (num_labels, hidden_size).

        Raises:
            NotImplementedError: If the model doesn't have a label encoder.
        """
        if self.config.labels_encoder is None:
            raise NotImplementedError("Labels pre-encoding is supported only for bi-encoder model.")

        # Create a DataLoader for efficient batching
        dataloader = DataLoader(labels, batch_size=batch_size, collate_fn=lambda x: x)

        labels_embeddings = []

        for batch in tqdm(dataloader, desc="Encoding labels"):
            tokenized_labels = self.data_processor.labels_tokenizer(
                batch, return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)
            with torch.no_grad():  # Disable gradient calculation for inference
                curr_labels_embeddings = self.model.token_rep_layer.encode_labels(**tokenized_labels)
            labels_embeddings.append(curr_labels_embeddings)

        return torch.cat(labels_embeddings, dim=0)

    @torch.no_grad()
    def batch_predict_with_embeds(
        self,
        texts: List[str],
        labels_embeddings: torch.Tensor,
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        packing_config: Optional[InferencePackingConfig] = None,
        input_spans: Optional[List[List[Dict]]] = None,
        return_class_probs: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities for a batch of texts using pre-computed label embeddings.

        Args:
            texts: A list of input texts to predict entities for.
            labels_embeddings: Pre-computed embeddings for the labels.
            labels: List of label strings corresponding to the embeddings.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            multi_label: Whether to allow multiple labels per token. Defaults to False.
            batch_size: Batch size for processing. Defaults to 8.
            packing_config: Configuration describing how to pack encoder inputs. When None
                the instance-level configuration set via configure_inference_packing is used.
            input_spans: Input entity spans to limit predictions to. Each span is a dict
                with 'start' and 'end' character positions.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.

        Returns:
            List of lists with predicted entities.
        """
        all_entities = self.inference(
            texts,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            batch_size=batch_size,
            packing_config=packing_config,
            input_spans=input_spans,
            return_class_probs=return_class_probs,
            labels_embeddings=labels_embeddings,
        )

        return all_entities

    def predict_with_embeds(
        self,
        text,
        labels_embeddings,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        return_class_probs=False,
        **kwargs,
    ):
        """Predict entities for a single text input using pre-computed label embeddings.

        Args:
            text: The input text to predict entities for.
            labels_embeddings: Pre-computed embeddings for the labels.
            labels: List of label strings corresponding to the embeddings.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            multi_label: Whether to allow multiple labels per entity. Defaults to False.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **kwargs: Additional arguments passed to batch_predict_with_embeds.

        Returns:
            List of entity predictions.
        """
        return self.batch_predict_with_embeds(
            [text],
            labels_embeddings,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            **kwargs,
        )[0]

    def _get_onnx_export_kwargs(self) -> dict[str, Any]:
        """Provide default labels for bi-encoder ONNX export."""
        return {"labels": ["organization", "person", "country"]}

    def _get_base_input_names(self) -> list[str]:
        """Get base input names (text-related inputs).

        Override in child classes for different architectures.
        """
        raise NotImplementedError

    def _get_label_input_names(self) -> list[str]:
        """Get label encoder input names."""
        return ["labels_input_ids", "labels_attention_mask"]

    def _get_embedding_input_name(self) -> str:
        """Get name for pre-computed embeddings input."""
        return "labels_embeddings"

    def _get_base_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Get base dynamic axes (text-related).

        Override in child classes for different architectures.
        """
        raise NotImplementedError

    def _get_label_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Get label encoder dynamic axes."""
        return {
            "labels_input_ids": {0: "num_labels", 1: "label_seq_length"},
            "labels_attention_mask": {0: "num_labels", 1: "label_seq_length"},
        }

    def _get_embedding_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Get pre-computed embeddings dynamic axes."""
        return {
            "labels_embeddings": {0: "num_labels", 1: "hidden_size"},
        }

    def _get_output_spec(self) -> dict[str, Any]:
        """Get output specification.

        Override in child classes for different output shapes.
        """
        raise NotImplementedError

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Build full input spec with label encoder.

        This is the default spec (without pre-computed embeddings).
        """
        base_names = self._get_base_input_names()
        label_names = self._get_label_input_names()

        base_axes = self._get_base_dynamic_axes()
        label_axes = self._get_label_dynamic_axes()
        output_spec = self._get_output_spec()

        return {
            "input_names": base_names + label_names,
            "output_names": ["logits"],
            "dynamic_axes": {**base_axes, **label_axes, **output_spec},
        }

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """
        Create flexible wrapper that handles both modes.

        The wrapper signature adapts based on input spec.
        """
        input_names = self._get_onnx_input_spec()["input_names"]

        class BiEncoderWrapper(nn.Module):
            def __init__(self, core, param_names):
                super().__init__()
                self.core = core
                self.param_names = param_names

            def forward(self, *args):
                # Build kwargs from positional args
                kwargs = dict(zip(self.param_names, args))
                out = self.core(**kwargs)
                return out.logits if hasattr(out, "logits") else out[0]

        return BiEncoderWrapper(core_model, input_names)

    def _prepare_onnx_batch(
        self,
        batch: Dict[str, torch.Tensor],
        from_labels_embeddings: bool = False,
        labels: Optional[list[str]] = None,
        **export_kwargs,
    ) -> tuple[tuple, Dict[str, Any]]:
        """
        Prepare batch for bi-encoder export with optional pre-computed embeddings.

        Args:
            batch: Dummy batch
            from_labels_embeddings: If True, use pre-computed embeddings mode
            labels: Labels for embedding computation
            **export_kwargs: Additional arguments

        Returns:
            Tuple of (inputs, spec)
        """
        if from_labels_embeddings:
            if not hasattr(self, "encode_labels"):
                raise RuntimeError("from_labels_embeddings=True requires encode_labels() method")

            # Compute embeddings
            if labels is None:
                labels = batch.get("labels", ["organization", "person", "country"])
            labels_embeds = self.encode_labels(labels).to("cpu")

            # Build spec with embeddings
            base_names = self._get_base_input_names()
            embed_name = self._get_embedding_input_name()

            base_axes = self._get_base_dynamic_axes()
            embed_axes = self._get_embedding_dynamic_axes()
            output_spec = self._get_output_spec()

            spec = {
                "input_names": [*base_names, embed_name],
                "output_names": ["logits"],
                "dynamic_axes": {**base_axes, **embed_axes, **output_spec},
            }

            # Build inputs
            all_inputs = tuple(labels_embeds if name == embed_name else batch[name] for name in spec["input_names"])

            return all_inputs, spec
        else:
            # Use default spec (full bi-encoder)
            return super()._prepare_onnx_batch(batch, **export_kwargs)


class UniEncoderSpanGLiNER(BaseEncoderGLiNER):
    config_class = UniEncoderSpanConfig
    model_class = UniEncoderSpanModel
    ort_model_class: type = UniEncoderSpanORTModel
    data_processor_class = UniEncoderSpanProcessor
    data_collator_class = UniEncoderSpanDataCollator
    decoder_class = SpanDecoder

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Define ONNX input specification for UniEncoderSpan model."""
        return {
            "input_names": [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
            ],
            "output_names": ["logits"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
                "span_mask": {0: "batch_size", 1: "num_spans"},
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_spans",
                    3: "num_classes",
                },
            },
        }

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """Create wrapper for UniEncoderSpan ONNX export."""

        class UniEncoderSpanWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(
                self,
                input_ids,
                attention_mask,
                words_mask,
                text_lengths,
                span_idx,
                span_mask,
            ):
                out = self.core(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    words_mask=words_mask,
                    text_lengths=text_lengths,
                    span_idx=span_idx,
                    span_mask=span_mask,
                )
                return out.logits if hasattr(out, "logits") else out[0]

        return UniEncoderSpanWrapper(core_model)


class UniEncoderTokenGLiNER(BaseEncoderGLiNER):
    config_class = UniEncoderTokenConfig
    model_class = UniEncoderTokenModel
    ort_model_class: type = UniEncoderTokenORTModel
    data_processor_class = UniEncoderTokenProcessor
    data_collator_class = UniEncoderTokenDataCollator
    decoder_class = TokenDecoder

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Define ONNX input specification for UniEncoderToken model."""
        return {
            "input_names": [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
            ],
            "output_names": ["logits"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "logits": {
                    0: "position",
                    1: "batch_size",
                    2: "sequence_length",
                    3: "num_classes",
                },
            },
        }

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """Create wrapper for UniEncoderToken ONNX export."""

        class UniEncoderTokenWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(
                self,
                input_ids,
                attention_mask,
                words_mask,
                text_lengths,
            ):
                out = self.core(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    words_mask=words_mask,
                    text_lengths=text_lengths,
                )
                return out.logits if hasattr(out, "logits") else out[0]

        return UniEncoderTokenWrapper(core_model)


class BiEncoderSpanGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderSpanConfig
    model_class = BiEncoderSpanModel
    ort_model_class: type = BiEncoderSpanORTModel
    data_processor_class = BiEncoderSpanProcessor
    data_collator_class = BiEncoderSpanDataCollator
    decoder_class = SpanDecoder

    def _get_base_input_names(self) -> list[str]:
        """Base inputs for span-based bi-encoder."""
        return [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        ]

    def _get_base_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Base dynamic axes for span-based bi-encoder."""
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
            "span_mask": {0: "batch_size", 1: "num_spans"},
        }

    def _get_output_spec(self) -> dict[str, Any]:
        """Output specification for span-based model."""
        return {
            "logits": {
                0: "batch_size",
                1: "sequence_length",
                2: "num_spans",
                3: "num_classes",
            },
        }


class BiEncoderTokenGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderTokenConfig
    model_class = BiEncoderTokenModel
    ort_model_class: type = BiEncoderTokenORTModel
    data_processor_class = BiEncoderTokenProcessor
    data_collator_class = BiEncoderTokenDataCollator
    decoder_class = TokenDecoder

    def _get_base_input_names(self) -> list[str]:
        """Base inputs for token-based bi-encoder."""
        return [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
        ]

    def _get_base_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """Base dynamic axes for token-based bi-encoder."""
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
        }

    def _get_output_spec(self) -> dict[str, Any]:
        """Output specification for token-based model."""
        return {
            "logits": {
                0: "position",
                1: "batch_size",
                2: "sequence_length",
                3: "num_classes",
            },
        }


class UniEncoderSpanDecoderGLiNER(BaseEncoderGLiNER):
    """GLiNER model with span-based encoding and label decoding capabilities.

    Supports generating textual labels for entities.
    """

    config_class = UniEncoderSpanDecoderConfig  # Uses base config with labels_decoder settings
    model_class = UniEncoderSpanDecoderModel
    ort_model_class: type = None
    data_processor_class = UniEncoderSpanDecoderProcessor
    data_collator_class = UniEncoderSpanDecoderDataCollator
    decoder_class = SpanGenerativeDecoder

    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create data processor with decoder tokenizer."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)

        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        # Load decoder tokenizer
        decoder_tokenizer = None
        if config.labels_decoder is not None:
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                config.labels_decoder, cache_dir=cache_dir, add_prefix_space=True
            )
            if decoder_tokenizer.pad_token is None:
                decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

        self.data_processor = self.data_processor_class(
            config, tokenizer, words_splitter, decoder_tokenizer=decoder_tokenizer
        )
        return self.data_processor

    def set_labels_trie(self, labels: List[str]):
        """Initialize the labels trie for constrained generation.

        Args:
            labels: Labels that will be used for constrained generation.

        Returns:
            Trie structure for constrained beam search.

        Raises:
            NotImplementedError: If the model doesn't have a decoder.
        """
        if self.data_processor.decoder_tokenizer is None:
            raise NotImplementedError("Label trie is implemented only for models with decoder.")

        tokenized_labels = []
        for label in labels:
            tokens = self.data_processor.decoder_tokenizer.encode(label)
            if tokens[0] == self.data_processor.decoder_tokenizer.bos_token_id:
                tokens = tokens[1:]
            tokens.append(self.data_processor.decoder_tokenizer.eos_token_id)
            tokenized_labels.append(tokens)

        trie = LabelsTrie(tokenized_labels)
        return trie

    def generate_labels(self, model_output, **gen_kwargs):
        """Generate textual class labels for each entity span.

        Args:
            model_output: Model output containing decoder_embedding and decoder_embedding_mask.
            **gen_kwargs: Generation parameters (max_new_tokens, temperature, etc.).

        Returns:
            List of generated label strings.
        """
        dec_embeds = model_output.decoder_embedding
        if dec_embeds is None:
            return []

        dec_mask = model_output.decoder_embedding_mask

        gen_ids = self.model.generate_labels(
            dec_embeds,
            dec_mask,
            max_new_tokens=gen_kwargs.pop("max_new_tokens", 15),
            eos_token_id=self.data_processor.decoder_tokenizer.eos_token_id,
            pad_token_id=self.data_processor.decoder_tokenizer.pad_token_id,
            do_sample=gen_kwargs.pop("do_sample", True),
            temperature=gen_kwargs.pop("temperature", 0.01),
            num_return_sequences=gen_kwargs.pop("num_return_sequences", 1),
            **gen_kwargs,
        )

        gen_texts = self.data_processor.decoder_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return gen_texts

    @torch.inference_mode()
    def run_batch(
        self,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        packing_config: Optional[InferencePackingConfig] = None,
        move_to_device: bool = True,
        gen_constraints: Optional[List[str]] = None,
        num_gen_sequences: int = 1,
        **gen_kwargs,
    ) -> Any:
        """Run model forward pass on a collated batch with label generation.

        Args:
            batch: Collated batch from collate_batch.
            threshold: Confidence threshold for predictions.
            packing_config: Optional inference packing configuration.
            move_to_device: Whether to move tensors to model device.
            gen_constraints: Labels to constrain generation.
            num_gen_sequences: Number of label sequences to generate per span.
            **gen_kwargs: Additional generation parameters.

        Returns:
            Model output with generated labels attached.
        """
        if move_to_device and not self.onnx_model:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        model_inputs = batch.copy() if packing_config is None else {**batch, "packing_config": packing_config}
        model_output = self.model(**model_inputs, threshold=threshold)

        # Generate labels if decoder is available
        gen_labels = None
        if self.config.labels_decoder is not None:
            labels_trie = self.set_labels_trie(gen_constraints) if gen_constraints else None
            gen_kwargs_copy = gen_kwargs.copy()
            gen_labels = self.generate_labels(
                model_output, labels_trie=labels_trie, num_return_sequences=num_gen_sequences, **gen_kwargs_copy
            )

        # Attach generated labels to model output for decode_batch
        model_output.gen_labels = gen_labels
        model_output.num_gen_sequences = num_gen_sequences

        return model_output

    def decode_batch(
        self,
        model_output: Any,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        input_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> List[List[Any]]:
        """Decode model output into entity predictions with generated labels.

        Args:
            model_output: Output from run_batch (includes gen_labels).
            batch: The collated batch (needs 'tokens' and 'id_to_classes').
            threshold: Confidence threshold for predictions.
            flat_ner: Whether to use flat NER (no overlapping entities).
            multi_label: Whether to allow multiple labels per span.
            return_class_probs: Whether to include class probabilities.
            input_spans: Optional word-level input spans to classify.

        Returns:
            List of entity lists (one per text in batch).
        """
        model_logits = model_output.logits
        if not isinstance(model_logits, torch.Tensor):
            model_logits = torch.from_numpy(model_logits)

        decoded = self.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            model_logits,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            gen_labels=model_output.gen_labels,
            sel_idx=model_output.decoder_span_idx,
            num_gen_sequences=model_output.num_gen_sequences,
            return_class_probs=return_class_probs,
            input_spans=input_spans,
        )
        return decoded

    def _process_batches(
        self,
        data_loader,
        threshold,
        flat_ner,
        multi_label,
        packing_config=None,
        return_class_probs=False,
        word_input_spans=None,
        gen_constraints=None,
        num_gen_sequences=1,
        **gen_kwargs,
    ):
        """Batch processing logic with label generation support."""
        outputs = []
        batch_offset = 0

        for batch in data_loader:
            model_output = self.run_batch(
                batch,
                threshold=threshold,
                packing_config=packing_config,
                move_to_device=True,
                gen_constraints=gen_constraints,
                num_gen_sequences=num_gen_sequences,
                **gen_kwargs,
            )

            batch_input_spans = None
            if word_input_spans is not None:
                current_batch_size = len(batch["tokens"])
                batch_input_spans = word_input_spans[batch_offset : batch_offset + current_batch_size]
                batch_offset += current_batch_size

            decoded = self.decode_batch(
                model_output,
                batch,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                input_spans=batch_input_spans,
            )
            outputs.extend(decoded)

        return outputs

    def map_entities_to_text(
        self,
        decoded: List[List[Any]],
        valid_texts: List[str],
        valid_to_orig_idx: List[int],
        start_token_map: List[List[int]],
        end_token_map: List[List[int]],
        num_original: int,
    ) -> List[List[Dict[str, Any]]]:
        """Map decoded entities back to character positions with generated labels.

        Args:
            decoded: Decoded entity spans from decode_batch.
            valid_texts: List of valid (non-empty) texts.
            valid_to_orig_idx: Mapping from valid indices to original indices.
            start_token_map: Per-text token-to-char-start mapping.
            end_token_map: Per-text token-to-char-end mapping.
            num_original: Total number of original texts.

        Returns:
            List of entity dicts aligned with original input texts.
        """
        all_entities = [[] for _ in range(num_original)]

        for valid_i, output in enumerate(decoded):
            orig_i = valid_to_orig_idx[valid_i]
            start_token_idx_to_text_idx = start_token_map[valid_i]
            end_token_idx_to_text_idx = end_token_map[valid_i]
            entities = []

            for span in output:
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]

                entity = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": valid_texts[valid_i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }

                if span.generated_labels is not None:
                    entity["generated_labels"] = span.generated_labels

                if span.class_probs is not None:
                    entity["class_probs"] = span.class_probs

                entities.append(entity)

            all_entities[orig_i] = entities

        return all_entities

    @torch.no_grad()
    def inference(
        self,
        texts: Union[str, List[str]],
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        batch_size: int = 8,
        gen_constraints: Optional[List[str]] = None,
        num_gen_sequences: int = 1,
        packing_config: Optional[InferencePackingConfig] = None,
        input_spans: Optional[List[List[Dict]]] = None,
        return_class_probs: bool = False,
        **gen_kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities with optional label generation.

        Args:
            texts: Input texts (string or list of strings).
            labels: Entity type labels.
            flat_ner: Whether to use flat NER.
            threshold: Confidence threshold.
            multi_label: Allow multiple labels per span.
            batch_size: Batch size for processing.
            gen_constraints: Labels to constrain generation.
            num_gen_sequences: Number of label sequences to generate per span.
            packing_config: Inference packing configuration.
            input_spans: Input entity spans to limit predictions to. Each span is a dict
                with 'start' and 'end' character positions.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **gen_kwargs: Additional generation parameters.

        Returns:
            List of entity predictions with optional generated labels.
        """
        self.eval()

        prepared = self.prepare_batch(texts, labels, input_spans)

        if not prepared["valid_texts"]:
            return [[] for _ in range(prepared["num_original"])]

        collator = self.create_collator()

        def collate_fn(batch):
            return self.collate_batch(batch, prepared["entity_types"], collator)

        data_loader = torch.utils.data.DataLoader(
            prepared["input_x"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        active_packing = packing_config if packing_config is not None else self._inference_packing_config

        outputs = self._process_batches(
            data_loader,
            threshold,
            flat_ner,
            multi_label,
            packing_config=active_packing,
            return_class_probs=return_class_probs,
            word_input_spans=prepared["word_input_spans"],
            gen_constraints=gen_constraints,
            num_gen_sequences=num_gen_sequences,
            **gen_kwargs,
        )

        all_entities = self.map_entities_to_text(
            outputs,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        return all_entities

    def predict_entities(
        self,
        text: str,
        labels: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        multi_label: bool = False,
        gen_constraints: Optional[List[str]] = None,
        num_gen_sequences: int = 1,
        return_class_probs: bool = False,
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """Predict entities for a single text input with optional label generation.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            multi_label: Whether to allow multiple labels per entity. Defaults to False.
            gen_constraints: Labels to constrain generation.
            num_gen_sequences: Number of label sequences to generate per span.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **gen_kwargs: Additional generation parameters.

        Returns:
            List of entity predictions as dictionaries.
        """
        return self.inference(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            gen_constraints=gen_constraints,
            num_gen_sequences=num_gen_sequences,
            return_class_probs=return_class_probs,
            **gen_kwargs,
        )[0]

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> dict[str, Optional[str]]:
        """
        ONNX export not supported for encoder-decoder models.

        Raises:
            NotImplementedError: Always raised as this model type cannot be exported to ONNX
        """
        raise NotImplementedError(
            "ONNX export is not supported for encoder-decoder GLiNER models "
            "(UniEncoderSpanDecoderGLiNER) because of the generative decoder head. "
            "The decoder requires iterative generation which is not suitable for "
            "static ONNX graph export. Consider:\n"
            "1. Export the encoder-only variant (UniEncoderSpanGLiNER)\n"
            "2. Use PyTorch for inference with this model\n"
            "3. Implement a custom ONNX pipeline with separate encoder/decoder exports"
        )


class UniEncoderTokenDecoderGLiNER(UniEncoderSpanDecoderGLiNER):
    """GLiNER model with token-based encoding and label decoding capabilities.

    Combines token-level BIO tagging with a decoder that generates entity type
    labels autoregressively.
    """

    config_class = UniEncoderTokenDecoderConfig
    model_class = UniEncoderTokenDecoderModel
    ort_model_class = None
    data_processor_class = UniEncoderTokenDecoderProcessor
    data_collator_class = UniEncoderTokenDecoderDataCollator
    decoder_class = TokenGenerativeDecoder


class UniEncoderSpanRelexGLiNER(BaseEncoderGLiNER):
    """GLiNER model for both entity recognition and relation extraction.

    Performs joint entity and relation prediction, allowing the model to simultaneously
    detect entities and the relationships between them in a single forward pass.
    """

    config_class = UniEncoderSpanRelexConfig
    model_class = UniEncoderSpanRelexModel
    ort_model_class: type = UniEncoderSpanRelexORTModel
    data_processor_class = RelationExtractionSpanProcessor
    data_collator_class = RelationExtractionSpanDataCollator
    decoder_class = SpanRelexDecoder

    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create relation extraction data processor."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
            self._set_tokenizer_spec_tokens(tokenizer)

        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter)
        return self.data_processor

    def _get_special_tokens(self):
        """Get special tokens to add to tokenizer.

        Can be overridden by child classes.

        Returns:
            List of special tokens
        """
        tokens = [self.config.ent_token, self.config.sep_token, self.config.rel_token]
        return tokens

    def set_class_indices(self):
        """Set the class token indices for entities and relations in the configuration."""
        self.config.class_token_index = len(self.data_processor.transformer_tokenizer) - 3

        self.config.rel_token_index = len(self.data_processor.transformer_tokenizer) - 1

    def prepare_batch(
        self,
        texts: Union[str, List[str]],
        labels: Union[str, List[str], List[List[str]]],
        input_spans: Optional[List[List[Dict]]] = None,
        relations: Optional[Union[str, List[str], List[List[str]]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare raw inputs for inference including relation types.

        Args:
            texts: Single text string or list of texts.
            labels: Entity labels - string, list of strings, or per-text label lists.
            input_spans: Optional pre-defined spans to classify (character positions).
            relations: Relation type labels - string, list of strings, or per-text label lists.
            **kwargs: Additional keyword arguments passed to the parent prepare_batch.

        Returns:
            Dictionary containing prepared inputs plus relation_types.
        """
        prepared = super().prepare_batch(texts, labels, input_spans)

        if relations is None:
            relation_types = []
        elif isinstance(relations, str):
            relation_types = list(dict.fromkeys([relations]))
        elif relations and isinstance(relations[0], list):
            relation_types = [list(dict.fromkeys(rels)) for rels in relations]
        else:
            relation_types = list(dict.fromkeys(relations))

        prepared["relation_types"] = relation_types

        return prepared

    def collate_batch(
        self,
        input_x: List[Dict[str, Any]],
        entity_types: Union[List[str], List[List[str]]],
        collator: Optional[Any] = None,
        relation_types: Optional[Union[List[str], List[List[str]]]] = None,
    ) -> Dict[str, Any]:
        """Collate prepared inputs into a tensor batch with relation types.

        Args:
            input_x: List of input dicts from prepare_batch.
            entity_types: Entity type labels.
            collator: Optional pre-created collator instance.
            relation_types: Relation type labels (list or per-text lists).

        Returns:
            Collated batch dictionary with tensors ready for the model.
        """
        if collator is None:
            collator = self.create_collator()

        if relation_types is None:
            relation_types = []

        batch = collator(input_x, entity_types=entity_types, relation_types=relation_types)
        return batch

    def create_collator(self) -> Any:
        """Create a data collator instance for relation extraction.

        Returns:
            Configured data collator instance.
        """
        return self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            return_rel_id_to_classes=True,
            prepare_labels=False,
        )

    @torch.inference_mode()
    def run_batch(
        self,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        adjacency_threshold: Optional[float] = None,
        packing_config: Optional[InferencePackingConfig] = None,
        move_to_device: bool = True,
        **external_inputs,
    ) -> Any:
        """Run model forward pass on a collated batch.

        Args:
            batch: Collated batch from collate_batch.
            threshold: Confidence threshold for predictions.
            adjacency_threshold: Threshold for adjacency matrix reconstruction.
            packing_config: Optional inference packing configuration.
            move_to_device: Whether to move tensors to model device.
            **external_inputs: Additional inputs to pass to the model.

        Returns:
            Model output containing logits and relation information.
        """
        if adjacency_threshold is None:
            adjacency_threshold = threshold

        if move_to_device and not self.onnx_model:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if packing_config is not None or external_inputs:
            model_inputs = {**batch, **external_inputs}
            if packing_config is not None:
                model_inputs["packing_config"] = packing_config
        else:
            model_inputs = batch

        model_output = self.model(**model_inputs, threshold=threshold, adjacency_threshold=adjacency_threshold)
        return model_output

    def decode_batch(
        self,
        model_output: Any,
        batch: Dict[str, Any],
        threshold: float = 0.5,
        relation_threshold: Optional[float] = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        input_spans: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Tuple[List[List[Any]], List[List[Any]]]:
        """Decode model output into entity and relation predictions.

        Args:
            model_output: Output from run_batch.
            batch: The collated batch.
            threshold: Confidence threshold for entity predictions.
            relation_threshold: Confidence threshold for relation predictions.
            flat_ner: Whether to use flat NER.
            multi_label: Whether to allow multiple labels per span.
            return_class_probs: Whether to include class probabilities.
            input_spans: Optional word-level input spans to classify.

        Returns:
            Tuple of (entity_outputs, relation_outputs) where each is a list per text.
        """
        if relation_threshold is None:
            relation_threshold = threshold

        model_logits = model_output.logits
        if not isinstance(model_logits, torch.Tensor):
            model_logits = torch.from_numpy(model_logits)

        rel_idx = model_output.rel_idx
        if not isinstance(rel_idx, torch.Tensor):
            rel_idx = torch.from_numpy(rel_idx)

        rel_logits = model_output.rel_logits
        if not isinstance(rel_logits, torch.Tensor):
            rel_logits = torch.from_numpy(rel_logits)

        rel_mask = model_output.rel_mask
        if not isinstance(rel_mask, torch.Tensor):
            rel_mask = torch.from_numpy(rel_mask)

        entity_spans = getattr(model_output, "entity_spans", None)
        if entity_spans is not None and not isinstance(entity_spans, torch.Tensor):
            entity_spans = torch.from_numpy(entity_spans)

        decoded_results = self.decoder.decode(
            batch["tokens"],
            batch["id_to_classes"],
            model_logits,
            rel_idx=rel_idx,
            rel_logits=rel_logits,
            rel_mask=rel_mask,
            flat_ner=flat_ner,
            threshold=threshold,
            relation_threshold=relation_threshold,
            multi_label=multi_label,
            rel_id_to_classes=batch["rel_id_to_classes"],
            entity_spans=entity_spans,
        )

        if len(decoded_results) == 2:
            decoded_entities, decoded_relations = decoded_results
        else:
            decoded_entities = decoded_results
            decoded_relations = [[] for _ in range(len(batch["tokens"]))]

        return decoded_entities, decoded_relations

    def _process_batches(
        self,
        data_loader,
        threshold,
        flat_ner,
        multi_label,
        packing_config=None,
        return_class_probs=False,
        word_input_spans=None,
        adjacency_threshold=None,
        relation_threshold=None,
        return_relations=True,
        **external_inputs,
    ):
        """Batch processing logic for entity and relation extraction."""
        all_entity_outputs = []
        all_relation_outputs = []
        batch_offset = 0

        for batch in data_loader:
            model_output = self.run_batch(
                batch,
                threshold=threshold,
                adjacency_threshold=adjacency_threshold,
                packing_config=packing_config,
                move_to_device=True,
                **external_inputs,
            )

            batch_input_spans = None
            if word_input_spans is not None:
                current_batch_size = len(batch["tokens"])
                batch_input_spans = word_input_spans[batch_offset : batch_offset + current_batch_size]
                batch_offset += current_batch_size

            decoded_entities, decoded_relations = self.decode_batch(
                model_output,
                batch,
                threshold=threshold,
                relation_threshold=relation_threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                return_class_probs=return_class_probs,
                input_spans=batch_input_spans,
            )

            all_entity_outputs.extend(decoded_entities)
            if return_relations:
                all_relation_outputs.extend(decoded_relations)
            else:
                for _ in range(len(batch["tokens"])):
                    all_relation_outputs.append(None)

        return all_entity_outputs, all_relation_outputs

    def map_entities_to_text(
        self,
        decoded: List[List[Any]],
        valid_texts: List[str],
        valid_to_orig_idx: List[int],
        start_token_map: List[List[int]],
        end_token_map: List[List[int]],
        num_original: int,
    ) -> List[List[Dict[str, Any]]]:
        """Map decoded entities back to character positions in original texts.

        Args:
            decoded: Decoded entity spans from decode_batch.
            valid_texts: List of valid (non-empty) texts.
            valid_to_orig_idx: Mapping from valid indices to original indices.
            start_token_map: Per-text token-to-char-start mapping.
            end_token_map: Per-text token-to-char-end mapping.
            num_original: Total number of original texts.

        Returns:
            List of entity dicts aligned with original input texts.
        """
        all_entities = [[] for _ in range(num_original)]

        for valid_i, output in enumerate(decoded):
            orig_i = valid_to_orig_idx[valid_i]
            start_token_idx_to_text_idx = start_token_map[valid_i]
            end_token_idx_to_text_idx = end_token_map[valid_i]
            entities = []

            for span in output:
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]

                entity = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": valid_texts[valid_i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }

                if span.class_probs is not None:
                    entity["class_probs"] = span.class_probs

                entities.append(entity)

            all_entities[orig_i] = entities

        return all_entities

    def map_relations_to_text(
        self,
        relation_outputs: List[List[Any]],
        entity_outputs: List[List[Any]],
        valid_texts: List[str],
        valid_to_orig_idx: List[int],
        start_token_map: List[List[int]],
        end_token_map: List[List[int]],
        num_original: int,
    ) -> List[List[Dict[str, Any]]]:
        """Map relation predictions back to character positions.

        Args:
            relation_outputs: Decoded relations per text.
            entity_outputs: Decoded entities per text (for getting span info).
            valid_texts: List of valid (non-empty) texts.
            valid_to_orig_idx: Mapping from valid indices to original indices.
            start_token_map: Per-text token-to-char-start mapping.
            end_token_map: Per-text token-to-char-end mapping.
            num_original: Total number of original texts.

        Returns:
            List of relation dicts aligned with original input texts.
        """
        return self._process_relations(
            relation_outputs,
            entity_outputs,
            start_token_map,
            end_token_map,
            valid_texts,
            valid_to_orig_idx,
            num_original,
        )

    @torch.no_grad()
    def inference(
        self,
        texts: Union[str, List[str]],
        labels: Union[str, List[str], List[List[str]]],
        relations: Union[str, List[str], List[List[str]]] = [],
        flat_ner: bool = True,
        threshold: float = 0.5,
        adjacency_threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        multi_label: bool = False,
        batch_size: int = 8,
        packing_config: Optional[InferencePackingConfig] = None,
        input_spans: Optional[List[List[Dict]]] = None,
        return_relations: bool = True,
        return_class_probs: bool = False,
    ) -> Union[List[List[Dict[str, Any]]], Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]]:
        """Predict entities and relations.

        Args:
            texts: Input texts (str or List[str]).
            labels: Entity type labels - string, list of strings, or per-text label lists.
            relations: Relation type labels - string, list of strings, or per-text label lists.
            flat_ner: Whether to use flat NER (no nested entities).
            threshold: Confidence threshold for entities.
            adjacency_threshold: Confidence threshold for adjacency matrix reconstruction (defaults to threshold).
            relation_threshold: Confidence threshold for relations (defaults to threshold).
            multi_label: Allow multiple labels per span.
            batch_size: Batch size for processing.
            packing_config: Inference packing configuration.
            input_spans: Input entity spans to limit predictions to. Each span is a dict
                with 'start' and 'end' character positions.
            return_relations: Whether to return relation predictions.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.

        Returns:
            Tuple of (entities, relations) if return_relations=True, else just entities.
        """
        self.eval()

        prepared = self.prepare_batch(texts, labels, input_spans, relations)

        if not prepared["valid_texts"]:
            if return_relations:
                return [[] for _ in range(prepared["num_original"])], [[] for _ in range(prepared["num_original"])]
            return [[] for _ in range(prepared["num_original"])]

        if relation_threshold is None:
            relation_threshold = threshold

        if adjacency_threshold is None:
            adjacency_threshold = threshold

        collator = self.create_collator()

        def collate_fn(batch):
            return self.collate_batch(batch, prepared["entity_types"], collator, prepared["relation_types"])

        data_loader = torch.utils.data.DataLoader(
            prepared["input_x"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        active_packing = packing_config if packing_config is not None else self._inference_packing_config

        entity_outputs, relation_outputs = self._process_batches(
            data_loader,
            threshold,
            flat_ner,
            multi_label,
            packing_config=active_packing,
            return_class_probs=return_class_probs,
            word_input_spans=prepared["word_input_spans"],
            adjacency_threshold=adjacency_threshold,
            relation_threshold=relation_threshold,
            return_relations=return_relations,
        )

        all_entities = self.map_entities_to_text(
            entity_outputs,
            prepared["valid_texts"],
            prepared["valid_to_orig_idx"],
            prepared["start_token_map"],
            prepared["end_token_map"],
            prepared["num_original"],
        )

        if return_relations:
            all_relations = self.map_relations_to_text(
                relation_outputs,
                entity_outputs,
                prepared["valid_texts"],
                prepared["valid_to_orig_idx"],
                prepared["start_token_map"],
                prepared["end_token_map"],
                prepared["num_original"],
            )
            return all_entities, all_relations

        return all_entities

    def predict_entities(
        self,
        text: str,
        labels: List[str],
        relations: List[str] = [],
        flat_ner: bool = True,
        threshold: float = 0.5,
        adjacency_threshold: Optional[float] = None,
        multi_label: bool = False,
        return_class_probs: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The entity labels to predict.
            relations: The relation labels (used for context but entities only returned).
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for predictions. Defaults to 0.5.
            adjacency_threshold: Threshold for adjacency matrix reconstruction. Defaults to threshold.
            multi_label: Whether to allow multiple labels per entity. Defaults to False.
            return_class_probs: Whether to include class probabilities in output. Defaults to False.
            **kwargs: Additional arguments passed to inference.

        Returns:
            List of entity predictions as dictionaries.
        """
        return self.inference(
            [text],
            labels,
            relations=relations,
            flat_ner=flat_ner,
            threshold=threshold,
            adjacency_threshold=adjacency_threshold,
            multi_label=multi_label,
            return_relations=False,
            return_class_probs=return_class_probs,
            **kwargs,
        )[0]

    def predict_relations(
        self,
        text: str,
        labels: List[str],
        relations: List[str],
        flat_ner: bool = True,
        threshold: float = 0.5,
        adjacency_threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        multi_label: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Predict entities and relations for a single text input.

        Args:
            text: The input text to predict entities and relations for.
            labels: The entity labels to predict.
            relations: The relation labels to predict.
            flat_ner: Whether to use flat NER. Defaults to True.
            threshold: Confidence threshold for entities. Defaults to 0.5.
            adjacency_threshold: Threshold for adjacency matrix reconstruction. Defaults to threshold.
            relation_threshold: Confidence threshold for relations. Defaults to threshold.
            multi_label: Whether to allow multiple labels per entity. Defaults to False.
            **kwargs: Additional arguments passed to inference.

        Returns:
            Tuple of (entities, relations) for the single text.
        """
        entities, rels = self.inference(
            [text],
            labels,
            relations=relations,
            flat_ner=flat_ner,
            threshold=threshold,
            adjacency_threshold=adjacency_threshold,
            relation_threshold=relation_threshold,
            multi_label=multi_label,
            return_relations=True,
            **kwargs,
        )
        return entities[0], rels[0]

    def _process_relations(
        self,
        relation_outputs,
        all_entity_outputs,
        all_start_token_idx_to_text_idx,
        all_end_token_idx_to_text_idx,
        valid_texts,
        valid_to_orig_idx=None,
        num_original_texts=None,
    ):
        """
        Process relation predictions into readable format.

        Args:
            relation_outputs: List of relation tuples per example, where each tuple is
                            (head_idx, relation_label, tail_idx, score)
            all_entity_outputs: List of entity outputs per example (token-level)
            all_start_token_idx_to_text_idx: Token to text index mappings (start)
            all_end_token_idx_to_text_idx: Token to text index mappings (end)
            valid_texts: Valid (non-empty) input texts
            valid_to_orig_idx: Mapping from valid index to original index (optional)
            num_original_texts: Total number of original texts (optional)

        Returns:
            List of relation lists, one per example
        """
        # If no mapping provided, assume 1:1 mapping
        if valid_to_orig_idx is None:
            valid_to_orig_idx = list(range(len(valid_texts)))
        if num_original_texts is None:
            num_original_texts = len(valid_texts)

        all_relations = [[] for _ in range(num_original_texts)]

        for valid_i, rel_tuples in enumerate(relation_outputs):
            orig_i = valid_to_orig_idx[valid_i]

            if rel_tuples is None or len(rel_tuples) == 0:
                continue

            relations = []
            entities_list = all_entity_outputs[valid_i]  # Token-level entities: (start, end, type, score)
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[valid_i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[valid_i]

            # Process each relation tuple from decoder
            for head_idx, relation_label, tail_idx, score in rel_tuples:
                # Validate entity indices
                if head_idx >= len(entities_list) or tail_idx >= len(entities_list):
                    continue

                # Get head and tail entities (using Span objects)
                head_span = entities_list[head_idx]
                tail_span = entities_list[tail_idx]

                # Convert token indices to text indices
                head_start_text = start_token_idx_to_text_idx[head_span.start]
                head_end_text = end_token_idx_to_text_idx[head_span.end]
                tail_start_text = start_token_idx_to_text_idx[tail_span.start]
                tail_end_text = end_token_idx_to_text_idx[tail_span.end]

                relations.append(
                    {
                        "head": {
                            "start": head_start_text,
                            "end": head_end_text,
                            "text": valid_texts[valid_i][head_start_text:head_end_text],
                            "type": head_span.entity_type,
                            "entity_idx": head_idx,
                        },
                        "tail": {
                            "start": tail_start_text,
                            "end": tail_end_text,
                            "text": valid_texts[valid_i][tail_start_text:tail_end_text],
                            "type": tail_span.entity_type,
                            "entity_idx": tail_idx,
                        },
                        "relation": relation_label,
                        "score": score,
                    }
                )

            all_relations[orig_i] = relations

        return all_relations

    @torch.no_grad()
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        flat_ner: bool = False,
        multi_label: bool = False,
        threshold: float = 0.5,
        adjacency_threshold: Optional[float] = None,
        relation_threshold: Optional[float] = None,
        batch_size: int = 12,
    ) -> Tuple[Tuple[Any, float], Tuple[Any, float]]:
        """Evaluate the model on both NER and relation extraction tasks.

        Args:
            test_data: The test data containing text, entity, and relation annotations.
            flat_ner: Whether to use flat NER. Defaults to False.
            multi_label: Whether to use multi-label classification. Defaults to False.
            threshold: The threshold for entity predictions. Defaults to 0.5.
            adjacency_threshold: Threshold for adjacency matrix reconstruction. Defaults to threshold.
            relation_threshold: The threshold for relation predictions. Defaults to threshold.
            batch_size: The batch size for evaluation. Defaults to 12.

        Returns:
            Tuple of ((ner_output, ner_f1), (rel_output, rel_f1)) containing:
            - ner_output: Formatted string with NER P, R, F1
            - ner_f1: NER F1 score
            - rel_output: Formatted string with relation extraction P, R, F1
            - rel_f1: Relation extraction F1 score
        """
        self.eval()

        if relation_threshold is None:
            relation_threshold = threshold

        if adjacency_threshold is None:
            adjacency_threshold = threshold

        # Create the dataset and data loader
        dataset = test_data
        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_relations=True,
            return_id_to_classes=True,
            return_rel_id_to_classes=True,
            prepare_labels=False,
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        all_entity_preds = []
        all_relation_preds = []
        all_true_entities = []
        all_true_relations = []

        # Iterate over data batches
        for batch in data_loader:
            if not self.onnx_model:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # noqa: PLW2901

            # Get model predictions
            model_inputs = batch.copy()
            model_output = self.model(**model_inputs, threshold=threshold, adjacency_threshold=adjacency_threshold)

            # Extract logits and relation outputs
            model_logits = model_output.logits
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)

            rel_idx = model_output.rel_idx
            if not isinstance(rel_idx, torch.Tensor):
                rel_idx = torch.from_numpy(rel_idx)

            rel_logits = model_output.rel_logits
            if not isinstance(rel_logits, torch.Tensor):
                rel_logits = torch.from_numpy(rel_logits)

            rel_mask = model_output.rel_mask
            if not isinstance(rel_mask, torch.Tensor):
                rel_mask = torch.from_numpy(rel_mask)

            entity_spans = getattr(model_output, "entity_spans", None)
            if entity_spans is not None and not isinstance(entity_spans, torch.Tensor):
                entity_spans = torch.from_numpy(entity_spans)

            # Decode predictions
            decoded_results = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_logits,
                rel_idx=rel_idx,
                rel_logits=rel_logits,
                rel_mask=rel_mask,
                flat_ner=flat_ner,
                threshold=threshold,
                relation_threshold=relation_threshold,
                multi_label=multi_label,
                rel_id_to_classes=batch["rel_id_to_classes"],
                entity_spans=entity_spans,
            )

            # Unpack results
            if len(decoded_results) == 2:
                decoded_entities, decoded_relations = decoded_results
            else:
                decoded_entities = decoded_results
                decoded_relations = [[] for _ in range(len(decoded_entities))]

            all_entity_preds.extend(decoded_entities)
            all_relation_preds.extend(decoded_relations)

            # Extract ground truth
            all_true_entities.extend(batch["entities"])
            all_true_relations.extend(batch.get("relations", [[] for _ in range(len(batch["entities"]))]))

        # Evaluate NER
        ner_evaluator = BaseNEREvaluator(all_true_entities, all_entity_preds)
        ner_output, ner_f1 = ner_evaluator.evaluate()

        # Evaluate Relations
        # Format data for relation evaluator: list of (entities, relations) tuples
        all_true_rel_data = list(zip(all_true_entities, all_true_relations))
        all_pred_rel_data = list(zip(all_entity_preds, all_relation_preds))

        rel_evaluator = BaseRelexEvaluator(all_true_rel_data, all_pred_rel_data)
        rel_output, rel_f1 = rel_evaluator.evaluate()

        return (ner_output, ner_f1), (rel_output, rel_f1)

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Define ONNX input specification for UniEncoderSpanRelex model."""
        return {
            "input_names": [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
            ],
            "output_names": ["logits", "rel_idx", "rel_logits", "rel_mask"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
                "span_mask": {0: "batch_size", 1: "num_spans"},
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_spans",
                    3: "num_ent_classes",
                },
                "rel_idx": {
                    0: "batch_size",
                    1: "num_pairs",
                    2: "pair_index",
                },
                "rel_logits": {
                    0: "batch_size",
                    1: "num_pairs",
                    2: "num_rel_classes",
                },
                "rel_mask": {
                    0: "batch_size",
                    1: "num_pairs",
                },
            },
        }

    def _get_onnx_export_kwargs(self) -> dict[str, Any]:
        """Provide default labels for relation extraction ONNX export."""
        return {"labels": ["head", "tail"]}

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """Create wrapper for UniEncoderSpanRelex ONNX export."""

        class UniEncoderSpanRelexWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(
                self,
                input_ids,
                attention_mask,
                words_mask,
                text_lengths,
                span_idx,
                span_mask,
            ):
                out = self.core(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    words_mask=words_mask,
                    text_lengths=text_lengths,
                    span_idx=span_idx,
                    span_mask=span_mask,
                )
                # Return all outputs for relation extraction
                return out.logits, out.rel_idx, out.rel_logits, out.rel_mask

        return UniEncoderSpanRelexWrapper(core_model)


class UniEncoderTokenRelexGLiNER(UniEncoderSpanRelexGLiNER):
    """GLiNER model for both entity recognition and relation extraction.

    Performs joint entity and relation prediction, allowing the model to simultaneously
    detect entities and the relationships between them in a single forward pass.
    """

    config_class = UniEncoderTokenRelexConfig
    model_class = UniEncoderTokenRelexModel
    ort_model_class: type = UniEncoderTokenRelexORTModel
    data_processor_class = RelationExtractionTokenProcessor
    data_collator_class = RelationExtractionTokenDataCollator
    decoder_class = TokenRelexDecoder

    def _get_onnx_input_spec(self) -> dict[str, Any]:
        """Define ONNX input specification for UniEncoderSpanRelex model."""
        return {
            "input_names": [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
            ],
            "output_names": ["logits", "rel_idx", "rel_logits", "rel_mask"],
            "dynamic_axes": {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_ent_classes",
                    3: "num_idx_classes",
                },
                "rel_idx": {
                    0: "batch_size",
                    1: "num_pairs",
                    2: "pair_index",
                },
                "rel_logits": {
                    0: "batch_size",
                    1: "num_pairs",
                    2: "num_rel_classes",
                },
                "rel_mask": {
                    0: "batch_size",
                    1: "num_pairs",
                },
            },
        }

    def _get_onnx_export_kwargs(self) -> dict[str, Any]:
        """Provide default labels for relation extraction ONNX export."""
        return {"labels": ["head", "tail"]}

    def _create_onnx_wrapper(self, core_model: nn.Module) -> nn.Module:
        """Create wrapper for UniEncoderSpanRelex ONNX export."""

        class UniEncoderTokenRelexWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(
                self,
                input_ids,
                attention_mask,
                words_mask,
                text_lengths,
            ):
                out = self.core(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    words_mask=words_mask,
                    text_lengths=text_lengths,
                )
                # Return all outputs for relation extraction
                return out.logits, out.rel_idx, out.rel_logits, out.rel_mask

        return UniEncoderTokenRelexWrapper(core_model)


class GLiNER(nn.Module, PyTorchModelHubMixin):
    """Meta GLiNER class that automatically instantiates the appropriate GLiNER variant.

    This class provides a unified interface for all GLiNER models, automatically switching to
    specialized model types based on the model configuration. It supports various NER architectures
    including uni-encoder, bi-encoder, decoder-based, and relation extraction models.

    The class automatically detects the model type based on:
        - span_mode: Token-level vs span-level
        - labels_encoder: Uni-encoder vs bi-encoder
        - labels_decoder: Standard vs decoder-based
        - relations_layer: NER-only vs joint entity-relation extraction

    Attributes:
        model: The loaded GLiNER model instance (automatically typed).
        config: Model configuration.
        data_processor: Data processor for the model.
        decoder: Decoder for predictions.

    Examples:
        Load a pretrained uni-encoder span model:
        >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

        Load a bi-encoder model:
        >>> model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")

        Load from local configuration:
        >>> config = GLiNERConfig.from_pretrained("config.json")
        >>> model = GLiNER.from_config(config)

        Initialize from scratch:
        >>> config = GLiNERConfig(model_name="microsoft/deberta-v3-small")
        >>> model = GLiNER(config)
    """

    def __init__(self, config: Union[str, Path, GLiNERConfig], **kwargs):
        """Initialize a GLiNER model with automatic type detection.

        This constructor determines the appropriate GLiNER variant based on the configuration
        and replaces itself with an instance of that variant.

        Args:
            config: Model configuration (GLiNERConfig object, path to config file, or dict).
            **kwargs: Additional arguments passed to the specific GLiNER variant.

        Examples:
            >>> config = GLiNERConfig(model_name="bert-base-cased")
            >>> model = GLiNER(config)
            >>> model = GLiNER("path/to/gliner_config.json")
        """
        super().__init__()

        # Load config if it's a path or dict
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = GLiNERConfig(**config_dict)
            else:
                raise FileNotFoundError(f"Config file not found: {config}")
        elif isinstance(config, dict):
            config = GLiNERConfig(**config)

        # Determine the appropriate GLiNER class based on config
        gliner_class = self._get_gliner_class(config)

        # Create instance of the appropriate class
        new_instance = gliner_class(config, **kwargs)

        # Replace this instance with the specific GLiNER variant
        self.__class__ = type(new_instance)
        self.__dict__ = new_instance.__dict__

    @staticmethod
    def _get_gliner_class(config: GLiNERConfig):
        """Determine the appropriate GLiNER class based on configuration."""
        is_token_level = config.span_mode == "token_level"
        has_labels_encoder = config.labels_encoder is not None
        has_labels_decoder = config.labels_decoder is not None
        has_relations = config.relations_layer is not None

        if has_relations:
            if is_token_level:
                return UniEncoderTokenRelexGLiNER
            else:
                return UniEncoderSpanRelexGLiNER

        if has_labels_decoder:
            if has_labels_encoder:
                warnings.warn(
                    "labels_encoder and labels_decoder are both set. "
                    "Using decoder model (labels_encoder will be ignored).",
                    stacklevel=2,
                )
            if is_token_level:
                return UniEncoderTokenDecoderGLiNER
            return UniEncoderSpanDecoderGLiNER

        if has_labels_encoder:
            if is_token_level:
                return BiEncoderTokenGLiNER
            else:
                return BiEncoderSpanGLiNER

        if is_token_level:
            return UniEncoderTokenGLiNER
        else:
            return UniEncoderSpanGLiNER

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = None,
        resize_token_embeddings: Optional[bool] = True,
        compile_torch_model: Optional[bool] = False,
        quantize: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        variant: Optional[str] = None,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """Load a pretrained GLiNER model with automatic type detection.

        This method loads the configuration, determines the appropriate GLiNER variant,
        and delegates to that variant's from_pretrained method.

        Args:
            model_id: Model identifier or local path.
            revision: Model revision.
            cache_dir: Cache directory.
            force_download: Force redownload.
            proxies: Proxy configuration.
            resume_download: Resume interrupted downloads.
            local_files_only: Only use local files.
            token: HF token for private repos.
            map_location: Device to map model to.
            strict: Enforce strict state_dict loading.
            load_tokenizer: Whether to load tokenizer.
            resize_token_embeddings: Whether to resize embeddings.
            compile_torch_model: Whether to compile with torch.compile.
            quantize: Only ``"int8"`` is accepted (int8 dynamic quantization: torchao
                on GPU, FBGEMM on CPU). For precision-only changes (fp16/bf16), use
                ``dtype=``. ``None`` to disable.
            dtype: Target floating-point dtype for the loaded weights (e.g.
                ``torch.bfloat16``, ``"bf16"``, ``"fp16"``). When set, weights
                are cast during the state-dict read so the fp32 copy is never
                fully materialized; prefer this over ``quantize`` for plain
                precision changes.
            variant: ``"fp16"`` / ``"bf16"`` to prefer
                ``model.{variant}.safetensors`` over the default fp32 file.
                Best-effort with graceful fallback: if the publisher uploaded
                the variant, only that file is fetched; if not, warns and
                falls back to fp32 + cast on read. See the base-class
                ``from_pretrained`` docstring for the full contract.
                ``None`` (default) preserves prior behavior.
            load_onnx_model: Whether to load ONNX model instead of PyTorch.
            onnx_model_file: Path to ONNX model file.
            max_length: Override max_length in config.
            max_width: Override max_width in config.
            post_fusion_schema: Override post_fusion_schema in config.
            _attn_implementation: Override attention implementation.
            **model_kwargs: Additional model initialization arguments.

        Returns:
            Appropriate GLiNER model instance.

        Examples:
            >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            >>> model = GLiNER.from_pretrained("knowledgator/gliner-bi-small-v1.0")
            >>> model = GLiNER.from_pretrained("path/to/local/model", quantize="int8")
            >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", dtype="bf16")
            >>> # If the repo publishes model.bf16.safetensors, download only that:
            >>> model = GLiNER.from_pretrained("org/gliner_bf16-v1", variant="bf16")
        """
        # Canonicalize variant up front so it can narrow the download. The
        # outer ``GLiNER`` class doesn't inherit from ``BaseGLiNER``; reuse
        # the helpers directly so behavior stays in lockstep.
        normalized_variant = BaseGLiNER._normalize_variant(variant)

        # dtype-vs-variant consistency check MUST run before the probe.
        # Otherwise, when the variant file is missing on the Hub,
        # ``_resolve_variant`` downgrades to ``None`` and the inner
        # ``from_pretrained``'s consistency check is skipped — silently
        # accepting a ``variant="bf16", dtype="fp16"`` mismatch instead of
        # raising as documented.
        torch_dtype = BaseGLiNER._parse_dtype(dtype)
        if normalized_variant is not None:
            variant_dtype = BaseGLiNER._VARIANT_TO_DTYPE[normalized_variant]
            if torch_dtype is None:
                torch_dtype = variant_dtype
                # Propagate the variant's dtype so the inner cast-on-read still
                # produces the requested precision after a fallback.
                dtype = variant_dtype
            elif torch_dtype != variant_dtype:
                raise ValueError(
                    f"variant={normalized_variant!r} requires dtype={variant_dtype}; "
                    f"got dtype={torch_dtype}. Drop dtype= to inherit from variant, "
                    f"or unset variant= to load the default file."
                )

        # Probe for availability and warn-and-fall-back to None if the variant
        # file isn't published. The inner from_pretrained will see model_dir
        # is already populated and skip its own probe — no double round-trip.
        normalized_variant = BaseGLiNER._resolve_variant(
            model_id,
            normalized_variant,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )

        model_dir = BaseGLiNER._download_model(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
            variant=normalized_variant,
        )

        # Load config to determine model type
        config_file = model_dir / "gliner_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config file found in {model_dir}")

        with open(config_file) as f:
            config_dict = json.load(f)

        config_dict.pop("model_type", None)

        config = GLiNERConfig(**config_dict)

        # Determine the appropriate class
        gliner_class = cls._get_gliner_class(config)

        logger.info("Loading the following GLiNER type: %s...", gliner_class)
        # Delegate to the specific class's from_pretrained method
        return gliner_class.from_pretrained(
            model_id=model_id,
            model_dir=model_dir,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            load_tokenizer=load_tokenizer,
            resize_token_embeddings=resize_token_embeddings,
            compile_torch_model=compile_torch_model,
            quantize=quantize,
            dtype=dtype,
            variant=normalized_variant,
            max_length=max_length,
            max_width=max_width,
            post_fusion_schema=post_fusion_schema,
            _attn_implementation=_attn_implementation,
            load_onnx_model=load_onnx_model,
            onnx_model_file=onnx_model_file,
            **model_kwargs,
        )

    @classmethod
    def from_config(
        cls,
        config: Union[GLiNERConfig, str, Path, dict],
        cache_dir: Optional[Union[str, Path]] = None,
        load_tokenizer: bool = True,
        resize_token_embeddings: bool = True,
        backbone_from_pretrained: bool = True,
        compile_torch_model: bool = False,
        quantize: Optional[str] = None,
        map_location: str = "cpu",
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """Create a GLiNER model from configuration.

        Args:
            config: Model configuration (GLiNERConfig object, path to config file, or dict).
            cache_dir: Cache directory for downloads.
            load_tokenizer: Whether to load tokenizer.
            resize_token_embeddings: Whether to resize token embeddings.
            backbone_from_pretrained: Whether to load the backbone encoder from pretrained weights.
            compile_torch_model: Whether to compile with torch.compile.
            quantize: Only ``"int8"`` is accepted (int8 dynamic quantization: torchao
                on GPU, FBGEMM on CPU). For precision-only changes (fp16/bf16), use
                ``dtype=``. ``None`` to disable.
            map_location: Device to map model to.
            max_length: Override max_length in config.
            max_width: Override max_width in config.
            post_fusion_schema: Override post_fusion_schema in config.
            _attn_implementation: Override attention implementation.
            **model_kwargs: Additional model initialization arguments.

        Returns:
            Initialized GLiNER model instance.

        Examples:
            >>> config = GLiNERConfig(model_name="microsoft/deberta-v3-small")
            >>> model = GLiNER.from_config(config)
            >>> model = GLiNER.from_config("path/to/gliner_config.json")
        """
        # Load config if needed
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                config_ = GLiNERConfig(**config_dict)
            else:
                raise FileNotFoundError(f"Config file not found: {config}")
        elif isinstance(config, dict):
            config_ = GLiNERConfig(**config)

        # Determine the appropriate class
        gliner_class = cls._get_gliner_class(config_)

        # Delegate to that class's load_from_config
        return gliner_class.load_from_config(
            config=config,
            cache_dir=cache_dir,
            load_tokenizer=load_tokenizer,
            resize_token_embeddings=resize_token_embeddings,
            backbone_from_pretrained=backbone_from_pretrained,
            compile_torch_model=compile_torch_model,
            quantize=quantize,
            map_location=map_location,
            max_length=max_length,
            max_width=max_width,
            post_fusion_schema=post_fusion_schema,
            _attn_implementation=_attn_implementation,
            **model_kwargs,
        )

    @property
    def model_map(self) -> dict[str, dict[str, Any]]:
        """Map configuration patterns to their corresponding GLiNER classes.

        Returns:
            Dictionary mapping model types to their classes and descriptions.
        """
        return {
            "gliner_uni_encoder_span": {
                "class": UniEncoderSpanGLiNER,
                "description": "Standard span-based NER with single encoder",
                "config": {
                    "span_mode": "span_level",
                    "labels_encoder": None,
                    "labels_decoder": None,
                    "relations_layer": None,
                },
            },
            "gliner_uni_encoder_token": {
                "class": UniEncoderTokenGLiNER,
                "description": "Token-level NER with single encoder",
                "config": {
                    "span_mode": "token_level",
                    "labels_encoder": None,
                    "labels_decoder": None,
                    "relations_layer": None,
                },
            },
            "gliner_bi_encoder_span": {
                "class": BiEncoderSpanGLiNER,
                "description": "Span-based NER with separate text and label encoders",
                "config": {
                    "span_mode": "span_level",
                    "labels_encoder": "required",
                    "labels_decoder": None,
                    "relations_layer": None,
                },
            },
            "gliner_bi_encoder_token": {
                "class": BiEncoderTokenGLiNER,
                "description": "Token-level NER with separate text and label encoders",
                "config": {
                    "span_mode": "token_level",
                    "labels_encoder": "required",
                    "labels_decoder": None,
                    "relations_layer": None,
                },
            },
            "gliner_uni_encoder_span_decoder": {
                "class": UniEncoderSpanDecoderGLiNER,
                "description": "Span-based NER with label generation decoder",
                "config": {"span_mode": "span_level", "labels_decoder": "required", "relations_layer": None},
            },
            "gliner_uni_encoder_token_decoder": {
                "class": UniEncoderTokenDecoderGLiNER,
                "description": "Token-level NER with label generation decoder",
                "config": {"span_mode": "token_level", "labels_decoder": "required", "relations_layer": None},
            },
            "gliner_uni_encoder_span_relex": {
                "class": UniEncoderSpanRelexGLiNER,
                "description": "Joint entity and relation extraction with single encoder",
                "config": {"span_mode": "span_level", "labels_encoder": None, "relations_layer": "required"},
            },
            "gliner_uni_encoder_token_relex": {
                "class": UniEncoderTokenRelexGLiNER,
                "description": "Joint entity and relation extraction with single encoder, token-level",
                "config": {"span_mode": "token_level", "labels_encoder": None, "relations_layer": "required"},
            },
        }

    def get_model_type(self) -> str:
        """
        Get the type of the current model instance.

        Returns:
            String identifier of the model type
        """
        class_name = self.__class__.__name__

        type_mapping = {
            "UniEncoderSpanGLiNER": "gliner_uni_encoder_span",
            "UniEncoderTokenGLiNER": "gliner_uni_encoder_token",
            "BiEncoderSpanGLiNER": "gliner_bi_encoder_span",
            "BiEncoderTokenGLiNER": "gliner_bi_encoder_token",
            "UniEncoderSpanDecoderGLiNER": "gliner_uni_encoder_span_decoder",
            "UniEncoderTokenDecoderGLiNER": "gliner_uni_encoder_token_decoder",
            "UniEncoderSpanRelexGLiNER": "gliner_uni_encoder_span_relex",
            "UniEncoderTokenRelexGLiNER": "gliner_uni_encoder_token_relex",
        }

        return type_mapping.get(class_name, "unknown")

    def __repr__(self) -> str:
        """String representation of the model."""
        model_type = self.get_model_type()
        model_info = self.model_map.get(model_type, {})
        description = model_info.get("description", "Unknown model type")

        return (
            f"{self.__class__.__name__}(\n"
            f"  type={model_type},\n"
            f"  description='{description}',\n"
            f"  config={self.config.__class__.__name__}\n"
            f")"
        )
