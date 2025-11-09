import json
import os
import re
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union, Type, Any
from abc import ABC, abstractmethod

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

from .config import (BaseGLiNERConfig, 
                     UniEncoderSpanConfig, 
                     UniEncoderTokenConfig,
                     BiEncoderSpanConfig,
                     BiEncoderTokenConfig,
                     UniEncoderSpanDecoderConfig,
                     UniEncoderSpanRelexConfig,
                     GLiNERConfig)
from .data_processing import (BaseProcessor, 
                              UniEncoderSpanProcessor, 
                              UniEncoderTokenProcessor, 
                              BiEncoderSpanProcessor, 
                              BiEncoderTokenProcessor,
                              RelationExtractionSpanProcessor,
                              UniEncoderSpanDecoderProcessor)
from .data_processing.collator import (UniEncoderSpanDataCollator,
                                       BiEncoderSpanDataCollator,
                                       UniEncoderSpanDecoderDataCollator,
                                       RelationExtractionSpanDataCollator,
                                       UniEncoderTokenDataCollator,
                                       BiEncoderTokenDataCollator)
from .data_processing.tokenizer import WordsSplitter
from .decoding import SpanDecoder, TokenDecoder, SpanGenerativeDecoder
from .decoding.trie import LabelsTrie
from .evaluation import BaseNEREvaluator
from .training import TrainingArguments, Trainer
from .modeling.base import (BaseModel, 
                            UniEncoderSpanModel, 
                            UniEncoderTokenModel,
                            BiEncoderSpanModel,
                            BiEncoderTokenModel,
                            UniEncoderSpanRelexModel,
                            UniEncoderSpanDecoderModel,
                            )
from .infer_packing import InferencePackingConfig
from .onnx.model import BaseORTModel, UniEncoderSpanORTModel, UniEncoderTokenORTModel, BiEncoderSpanORTModel, BiEncoderTokenORTModel, UniEncoderSpanRelexORTModel
from .utils import is_module_available

if is_module_available("onnxruntime"):
    import onnxruntime as ort
    ONNX_AVAILABLE = True
else:
    ONNX_AVAILABLE = False

class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
    config_class: Type = None 
    model_class: Type = None
    ort_model_class: Type = None
    data_processor_class: Type = None
    data_collator_class: Type = None
    decoder_class: Type = None

    def __init__(
        self,
        config: BaseGLiNERConfig,
        model: Optional[BaseModel] = None,
        tokenizer: Optional[BaseModel] = None,
        data_processor: Optional[BaseProcessor] = None,
        backbone_from_pretrained: Optional[bool] = False,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
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
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        if self.onnx_model:
            providers = self.model.session.get_providers()
            if 'CUDAExecutionProvider' in providers:
                return torch.device('cuda')
            return torch.device('cpu')
        device = next(self.model.parameters()).device
        return device

    def configure_inference_packing(
        self, config: Optional[InferencePackingConfig]
    ) -> None:
        """Configure default packing behaviour for inference calls.

        Passing ``None`` disables packing by default. Individual inference
        methods accept a ``packing_config`` argument to override this setting
        on a per-call basis.
        """

        self._inference_packing_config = config

    def compile(self):
        self.model = torch.compile(self.model)

    def _get_special_tokens(self):
        """
        Get special tokens to add to tokenizer.
        Can be overridden by child classes.
        
        Returns:
            List of special tokens
        """
        tokens = ["[FLERT]", self.config.ent_token, self.config.sep_token]
        return tokens
    
    def prepare_state_dict(self, state_dict):
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[key] = tensor
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
        """
        Save model weights and configuration to local directory.

        Args:
            save_directory: Path to directory for saving
            config: Model configuration (uses self.config if None)
            repo_id: Repository ID for hub upload
            push_to_hub: Whether to push to HuggingFace Hub
            safe_serialization: Whether to use safetensors format
            **push_to_hub_kwargs: Additional arguments for push_to_hub
            
        Returns:
            Repository URL if pushed to hub, None otherwise
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
        with open(config_file, "r") as f:
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
    
    @classmethod
    def _load_tokenizer(cls, model_dir: Path, cache_dir: Optional[Path] = None):
        """
        Load tokenizer from directory.
        
        Args:
            model_dir: Directory containing tokenizer files
            cache_dir: Cache directory for downloads
            
        Returns:
            Tokenizer instance or None
        """
        if os.path.exists(model_dir / "tokenizer_config.json"):
            return AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        return None
    
    @classmethod
    def _load_state_dict(cls, model_file: Path, map_location: str = "cpu"):
        """
        Load state dict from file.
        
        Args:
            model_file: Path to model file
            map_location: Device to map tensors to
            
        Returns:
            State dict
        """
        if model_file.suffix == ".safetensors" or str(model_file).endswith(".safetensors"):
            state_dict = {}
            with safe_open(model_file, framework="pt", device=map_location) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(
                model_file, 
                map_location=torch.device(map_location), 
                weights_only=True
            )
        return state_dict
    
    @classmethod
    def _download_model(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        token: Union[str, bool, None] = None,
        local_files_only: bool = False,
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
            
        Returns:
            Path to model directory
        """
        model_dir = Path(model_id)
        
        if not model_dir.exists():
            model_dir = Path(snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            ))
        
        return model_dir
    
    @classmethod
    def load_from_config(
        cls,
        config: Union[str, Path, GLiNERConfig, Dict],
        cache_dir: Optional[Union[str, Path]] = None,
        load_tokenizer: bool = True,
        resize_token_embeddings: bool = True,
        backbone_from_pretrained: bool = True,
        compile_torch_model: bool = False,
        map_location: str = "cpu",
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Initialize a model from configuration without loading pretrained weights.
        
        This method creates a new model instance from scratch using the provided configuration.
        The backbone encoder can optionally be loaded from pretrained weights, but the GLiNER-specific
        layers are always randomly initialized.
        
        Args:
            config: Model configuration (GLiNERConfig object, path to config file, or dict)
            cache_dir: Cache directory for downloads
            load_tokenizer: Whether to load tokenizer
            resize_token_embeddings: Whether to resize token embeddings
            backbone_from_pretrained: Whether to load the backbone encoder from pretrained weights
            compile_torch_model: Whether to compile with torch.compile
            map_location: Device to map model to
            max_length: Override max_length in config
            max_width: Override max_width in config
            post_fusion_schema: Override post_fusion_schema in config
            _attn_implementation: Override attention implementation
            **model_kwargs: Additional model initialization arguments
            
        Returns:
            Initialized model instance with randomly initialized weights (except backbone if specified)
            
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
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config_dict.pop("model_type", None)
        elif isinstance(config, dict):
            config_dict = config.copy()
            config_dict.pop("model_type", None)
        elif isinstance(config, BaseGLiNERConfig):
            config_dict = config.to_dict()
        else:
            raise TypeError(
                f"config must be a GLiNERConfig object, path to config file, or dict. "
                f"Got {type(config)}"
            )
        
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
            tokenizer = AutoTokenizer.from_pretrained(
                config_instance.model_name, 
                cache_dir=cache_dir
            )
        
        # Create model instance from scratch
        instance = cls(
            config_instance,
            tokenizer=tokenizer,
            backbone_from_pretrained=backbone_from_pretrained,
            cache_dir=cache_dir,
            **model_kwargs
        )
        
        # Resize token embeddings if needed
        if resize_token_embeddings and (
            config_instance.class_token_index == -1 or config_instance.vocab_size == -1
        ):
            add_tokens = instance._get_special_tokens()
            instance.resize_embeddings(add_tokens=add_tokens)

            if tokenizer is not None:
                tokenizer.add_tokens(add_tokens, special_tokens=True)
                
        # Move to device
        instance.model.to(map_location)
        
        # Compile if requested
        if compile_torch_model:
            if "cuda" in map_location:
                print("Compiling torch model...")
                instance.compile()
            else:
                warnings.warn(
                    "Cannot compile model on CPU. Set `map_location='cuda'` to compile."
                )
        
        instance.eval()
        return instance

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = None,
        resize_token_embeddings: Optional[bool] = True,
        compile_torch_model: Optional[bool] = False,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        session_options = None,
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Load pretrained model from HuggingFace Hub or local directory.

        Args:
            model_id: Model identifier or local path
            revision: Model revision
            cache_dir: Cache directory
            force_download: Force redownload
            proxies: Proxy configuration
            resume_download: Resume interrupted downloads
            local_files_only: Only use local files
            token: HF token for private repos
            map_location: Device to map model to
            strict: Enforce strict state_dict loading
            load_tokenizer: Whether to load tokenizer
            resize_token_embeddings: Whether to resize embeddings
            compile_torch_model: Whether to compile with torch.compile
            max_length: Override max_length in config
            max_width: Override max_width in config
            post_fusion_schema: Override post_fusion_schema in config
            _attn_implementation: Override attention implementation
            **model_kwargs: Additional model initialization arguments
            
        Returns:
            Model instance
        """
        # Download or locate model
        model_dir = cls._download_model(
            model_id, revision, cache_dir, force_download,
            proxies, resume_download, token, local_files_only
        )
        
        # Find model file
        model_file = model_dir / "model.safetensors"
        if not model_file.exists():
            model_file = model_dir / "pytorch_model.bin"
        
        if not model_file.exists():
            raise FileNotFoundError(f"No model file found in {model_dir}")
        
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
            tokenizer = cls._load_tokenizer(model_dir, cache_dir)
        
        if not load_onnx_model:
            # Create model instance
            instance = cls(
                config,
                tokenizer=tokenizer,
                backbone_from_pretrained=False,
                cache_dir=cache_dir,
                **model_kwargs
            )
            
            # Resize token embeddings if needed
            add_tokens = instance._get_special_tokens()
            if resize_token_embeddings and (
                config.class_token_index == -1 or config.vocab_size == -1
            ):
                instance.resize_embeddings(add_tokens=add_tokens)
                instance.data_processor.transformer_tokenizer.add_tokens(add_tokens)

            # Load state dict
            state_dict = cls._load_state_dict(model_file, map_location)
            instance.model.load_state_dict(state_dict, strict=strict)
            instance.model.to(map_location)
            
            if compile_torch_model:
                if "cuda" in map_location:
                    print("Compiling torch model...")
                    instance.compile()
                else:
                    warnings.warn(
                        "Cannot compile model on CPU. Set `map_location='cuda'` to compile."
                    )
            
            instance.eval()
        else:
            model_file = Path(onnx_model_file)
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"The ONNX model can't be loaded from {model_file}."
                )
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
            providers = ['CPUExecutionProvider']
            if "cuda" in map_location:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available but `map_location` is set to 'cuda'.")
                providers = ['CUDAExecutionProvider']
            ort_session = ort.InferenceSession(model_file, session_options, providers=providers)
            model = cls.ort_model_class(ort_session)
            instance = cls(config, tokenizer=tokenizer, model=model)

        return instance

    @abstractmethod
    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        """
        Export this GLiNER variant to ONNX.

        Each concrete GLiNER subclass must implement its own variant-specific
        exporter that:
        - Uses the correct input signature for its architecture.
        - Produces outputs compatible with the corresponding ORTModel wrapper.
        - Optionally creates a dynamically quantized version.

        Returns:
            {
                "onnx_path": str,
                "quantized_path": Optional[str],
            }
        """
        raise NotImplementedError
    
    def _check_onnx_export_preconditions(self):
        if self.onnx_model:
            raise RuntimeError(
                "This instance already wraps an ONNX/ORT model. "
                "Export is intended for PyTorch-based models."
            )
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not available. Install `onnxruntime` to export to ONNX."
            )
        if not hasattr(self, "data_processor") or not hasattr(self, "data_collator_class"):
            raise RuntimeError(
                "Model is not fully initialized (missing data_processor or data_collator)."
            )

    def _maybe_import_quantization(self):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            return quantize_dynamic, QuantType
        except Exception:
            return None, None

    def _build_dummy_batch(
        self,
        labels: Optional[List[str]] = None,
        text: str = "ONNX export dummy input.",
    ) -> Dict[str, torch.Tensor]:
        """
        Build a single CPU batch using the model's own preprocessing stack.

        Concrete exporters can call this and then select the keys they need.
        """
        if labels is None or len(labels) == 0:
            labels = ["person", 'organization', 'country']

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

        print(batch.keys())
        return batch

    def _run_torch_onnx_export(
        self,
        wrapper: nn.Module,
        all_inputs: tuple,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
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
            dynamo=False
        )

    def _maybe_quantize_onnx(
        self,
        onnx_path: Path,
        quantized_path: Path,
        quantize: bool,
    ) -> Optional[Path]:
        if not quantize:
            return None

        quantize_dynamic, QuantType = self._maybe_import_quantization()
        if quantize_dynamic is None:
            warnings.warn(
                "onnxruntime.quantization is not available; skipping quantization."
            )
            return None

        try:
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(quantized_path),
                weight_type=QuantType.QUInt8,
            )
            return quantized_path
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}")
            return None
        
    def _create_data_collator(self, **kwargs):
        """
        Create data collator. Override in child classes if needed.
        
        Returns:
            Data collator instance
        """
        return self.data_collator_class(
                self.config, 
                data_processor=self.data_processor, 
                prepare_labels=True,
                **kwargs
            )
    
    def _get_freezable_components(self):
        """
        Get dictionary mapping component names to their actual modules.
        Returns:
            dict: Mapping of component names to module objects
        """
        components = {}
        
        # Text encoder (always present)
        if hasattr(self, 'model') and hasattr(self.model, 'token_rep_layer'):
            if hasattr(self.model.token_rep_layer, 'bert_layer'):
                components['text_encoder'] = self.model.token_rep_layer.bert_layer.model
        
        # Labels encoder (optional)
        if self.config.labels_encoder is not None:
            if hasattr(self.model, 'token_rep_layer') and hasattr(self.model.token_rep_layer, 'labels_encoder'):
                components['labels_encoder'] = self.model.token_rep_layer.labels_encoder.model
        
        # Decoder (optional)
        if self.config.labels_decoder is not None:
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'decoder_layer'):
                components['decoder'] = self.model.decoder.decoder_layer.model
        
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
            print(f"Frozen: {component_name}")
        else:
            available = ', '.join(components.keys())
            print(f"Warning: Component '{component_name}' not found. Available components: {available}")


    def unfreeze_component(self, component_name: str):
        """
        Unfreeze a specific component of the model.
        Args:
            component_name: Name of component to unfreeze
        """
        components = self._get_freezable_components()
        if component_name in components:
            components[component_name].requires_grad_(True)
            print(f"Unfrozen: {component_name}")
        else:
            available = ', '.join(components.keys())
            print(f"Warning: Component '{component_name}' not found. Available components: {available}")

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
        focal_loss_prob_margin: float = 0.0,
        loss_reduction: str = 'sum',
        negatives: float = 1.0,
        masking: str = 'none',
        lr_scheduler_type: str = 'linear',
        warmup_ratio: float = 0.1,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        max_grad_norm: float = 1.0,
        max_steps: int = 10000,
        save_steps: int = 1000,
        save_total_limit: int = 10,
        logging_steps: int = 10,
        use_cpu: bool = False,
        bf16: bool = True,
        dataloader_num_workers: int = 1,
        report_to: str = "none",
        **kwargs
    ) -> TrainingArguments:
        """
        Create training arguments with sensible defaults.
        
        Args:
            output_dir: Directory to save model checkpoints
            learning_rate: Learning rate for main parameters
            weight_decay: Weight decay for main parameters
            others_lr: Learning rate for other parameters
            others_weight_decay: Weight decay for other parameters
            focal_loss_alpha: Alpha for focal loss
            focal_loss_gamma: Gamma for focal loss
            focal_loss_prob_margin: Probability margin for focal loss
            loss_reduction: Loss reduction method
            negatives: Negative sampling ratio
            masking: Masking strategy
            lr_scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio
            per_device_train_batch_size: Training batch size
            per_device_eval_batch_size: Evaluation batch size
            max_grad_norm: Maximum gradient norm
            max_steps: Maximum training steps
            save_steps: Save checkpoint every N steps
            save_total_limit: Maximum number of checkpoints to keep
            logging_steps: Log every N steps
            use_cpu: Whether to use CPU
            bf16: Whether to use bfloat16
            dataloader_num_workers: Number of dataloader workers
            report_to: Where to report metrics
            **kwargs: Additional training arguments
            
        Returns:
            TrainingArguments instance
        """
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            others_lr=others_lr or learning_rate,
            others_weight_decay=others_weight_decay or weight_decay,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
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
            **kwargs
        )
    
    def train_model(
        self,
        train_dataset,
        eval_dataset,
        training_args: Optional[TrainingArguments] = None,
        freeze_components: Optional[List[str]] = None,
        compile_model: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        **training_kwargs
    ) -> Trainer:
        """
        Train the model.
        
        Args:
            train_data: Training data (path to JSON file or list of samples)
            training_args: Training arguments (created with defaults if None)
            eval_data: Evaluation data (optional, uses test_split if None)
            test_split: Fraction of train_data to use for eval if eval_data is None
            freeze_components: List of component names to freeze (e.g., ['text_encoder', 'decoder'])
            compile_model: Whether to compile model with torch.compile
            use_new_data_schema: Whether to use new data schema
            output_dir: Output directory (required if training_args is None)
            **training_kwargs: Additional kwargs for creating training args
            
        Returns:
            Trained Trainer instance
        """ 
        # Create training arguments if not provided
        if training_args is None:
            if output_dir is None:
                raise ValueError("Either training_args or output_dir must be provided")
            training_args = self.create_training_args(
                output_dir=output_dir,
                **training_kwargs
            )
        
        # Compile model if requested
        if compile_model:
            self.compile()
        
        # Freeze components if specified
        if freeze_components:
            for component_name in freeze_components:
                self.freeze_component(component_name)
        
        # Create data collator
        data_collator = self._create_data_collator()
        
        # Create trainer
        trainer = Trainer(
            model=self,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.data_processor.transformer_tokenizer,
            data_collator=data_collator,
        )
        
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
        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter)
        return self.data_processor

    def resize_embeddings(self):
        if (len(self.data_processor.transformer_tokenizer)!=self.config.vocab_size
                                                        and self.config.vocab_size!=-1):
            new_num_tokens = len(self.data_processor.transformer_tokenizer)
            model_embeds = self.model.token_rep_layer.resize_token_embeddings(
                new_num_tokens, None
            )

    def prepare_inputs(self, texts: List[str]):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
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
    
    def prepare_base_input(self, all_tokens):
        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        return input_x

    def _process_batches(self, data_loader, threshold, flat_ner, multi_label, packing_config=None):
        """Shared batch processing logic"""
        outputs = []
        is_onnx = self.onnx_model
        device = self.device
        
        for batch in data_loader:
            # Move to device once (outside condition)
            if not is_onnx:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Prepare model inputs
            model_inputs = batch.copy() if packing_config is None else {**batch, "packing_config": packing_config}
            
            # Get predictions
            model_logits = self.model(**model_inputs, threshold=threshold)[0]
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)

            # Decode
            decoded = self.decoder.decode(
                batch["tokens"], batch["id_to_classes"], model_logits,
                flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
            )
            outputs.extend(decoded)
        
        return outputs

    @torch.no_grad()
    def inference(
        self,
        texts,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        batch_size=8,
        packing_config: Optional[InferencePackingConfig] = None,
    ):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.
            packing_config (Optional[InferencePackingConfig], optional):
                Configuration describing how to pack encoder inputs. When ``None``
                the instance-level configuration set via
                :meth:`configure_inference_packing` is used.

        Returns:
            The list of lists with predicted entities.
        """
        self.eval()
        # raw input preparation
        if isinstance(texts, str):
            texts = [texts]

        entity_types = list(dict.fromkeys(labels))

        tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_inputs(texts)
        
        input_x = self.prepare_base_input(tokens)

        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        def collate_fn(batch, entity_types=entity_types):
            batch_out = collator(batch, entity_types=entity_types)
            return batch_out

        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        active_packing = (
            packing_config
            if packing_config is not None
            else self._inference_packing_config
        )
        outputs = self._process_batches(data_loader, threshold, flat_ner, multi_label, 
                                                        packing_config=active_packing)

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                ent_details =  {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                entities.append(ent_details)

            all_entities.append(entities)

        return all_entities
    
    def predict_entities(
        self, text, labels, flat_ner=True, threshold=0.5, multi_label=False, **kwargs
    ):
        """
        Predict entities for a single text input.

        Args:
            text: The input text to predict entities for.
            labels: The labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.inference(
            [text],
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs
        )[0]

    def batch_predict_entities(
        self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False, **kwargs
    ):
        """
        DEPRECATED: Use `run` instead.

        This method will be removed in a future release. It now forwards to
        `GLiNER.run(...)` to perform inference.

        Args:
            texts (List[str]): Input texts.
            labels (List[str]): Labels to predict.
            flat_ner (bool, optional): Use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold. Defaults to 0.5.
            multi_label (bool, optional): Allow multiple labels per token/entity. Defaults to False.
            **kwargs: Extra arguments forwarded to `run` (e.g., batch_size).
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
    
    def evaluate(
        self,
        test_data,
        flat_ner=False,
        multi_label=False,
        threshold=0.5,
        batch_size=12,
        entity_types=None,
    ):
        """
        Evaluate the model on a given test dataset.

        Args:
            test_data (List[Dict]): The test data containing text and entity annotations.
            flat_ner (bool): Whether to use flat NER. Defaults to False.
            multi_label (bool): Whether to use multi-label classification. Defaults to False.
            threshold (float): The threshold for predictions. Defaults to 0.5.
            batch_size (int): The batch size for evaluation. Defaults to 12.
            entity_types (Optional[List[str]]): List of entity types to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the evaluation output and the F1 score.
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
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        all_preds = self._process_batches(data_loader, threshold, flat_ner, multi_label)
        all_trues = []

        # Iterate over data batches
        for batch in data_loader:
            all_trues.extend(batch["entities"])

        # Evaluate the predictions
        evaluator = BaseNEREvaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()

        return out, f1
    
    
class BaseBiEncoderGLiNER(BaseEncoderGLiNER):
    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder, cache_dir=cache_dir)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)

        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter, labels_tokenizer=labels_tokenizer)
        return self.data_processor
    
    def resize_embeddings(self, **kwargs):
        warnings.warn("Resizing embeddings is not supported for bi-encoder models.")

    @torch.no_grad()
    def encode_labels(self, labels: List[str], batch_size: int = 8) -> torch.FloatTensor:
        """
        Embedding of labels.

        Args:
            labels (List[str]): A list of labels.
            batch_size (int): Batch size for processing labels.

        Returns:
            labels_embeddings (torch.FloatTensor): Tensor containing label embeddings.
        """
        if self.config.labels_encoder is None:
            raise NotImplementedError("Labels pre-encoding is supported only for bi-encoder model.")

        # Create a DataLoader for efficient batching
        dataloader = DataLoader(labels, batch_size=batch_size, collate_fn=lambda x: x)

        labels_embeddings = []

        for batch in tqdm(dataloader, desc="Encoding labels"):
            tokenized_labels = self.data_processor.labels_tokenizer(batch, return_tensors='pt',
                                                                truncation=True, padding="max_length").to(self.device)
            with torch.no_grad():  # Disable gradient calculation for inference
                curr_labels_embeddings = self.model.token_rep_layer.encode_labels(**tokenized_labels)
            labels_embeddings.append(curr_labels_embeddings)

        return torch.cat(labels_embeddings, dim=0)
    
    @torch.no_grad()
    def batch_predict_with_embeds(
        self,
        texts,
        labels_embeddings,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        batch_size=8,
        packing_config: Optional[InferencePackingConfig] = None,
    ):
        """
        Predict entities for a batch of texts using pre-computed label embeddings.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels_embeddings: Pre-computed embeddings for the labels.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.
            batch_size (int, optional): Batch size for processing. Defaults to 8.
            packing_config (Optional[InferencePackingConfig], optional):
                Configuration describing how to pack encoder inputs. When ``None``
                the instance-level configuration set via
                :meth:`configure_inference_packing` is used.

        Returns:
            The list of lists with predicted entities.
        """
        self.eval()
        
        # Raw input preparation
        if isinstance(texts, str):
            texts = [texts]

        tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_inputs(texts)
        
        input_x = self.prepare_base_input(tokens)

        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )
    
        def collate_fn(batch):
            batch_out = collator(batch)
            return batch_out
        
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        active_packing = (
            packing_config
            if packing_config is not None
            else self._inference_packing_config
        )

        # Process batches with embeddings
        outputs = []
        is_onnx = self.onnx_model
        device = self.device
        
        for batch in data_loader:
            # Move to device once (outside condition)
            if not is_onnx:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Prepare model inputs with labels_embeddings
            model_inputs = batch.copy() if active_packing is None else {**batch, "packing_config": active_packing}
            model_inputs["labels_embeddings"] = labels_embeddings
            
            # Get predictions
            model_logits = self.model(**model_inputs, threshold=threshold)[0]
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)
            
            # Decode
            decoded = self.decoder.decode(
                batch["tokens"], batch["id_to_classes"], model_logits,
                flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
            )
            outputs.extend(decoded)

        # Convert outputs to entities with text indices
        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append(
                    {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                )
            all_entities.append(entities)

        return all_entities

    def predict_with_embeds(
        self, text, labels_embeddings, labels, flat_ner=True, threshold=0.5, multi_label=False, **kwargs
    ):
        """
        Predict entities for a single text input using pre-computed label embeddings.

        Args:
            text: The input text to predict entities for.
            labels_embeddings: Pre-computed embeddings for the labels.
            labels: The labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per entity. Defaults to False.

        Returns:
            The list of entity predictions.
        """
        return self.batch_predict_with_embeds(
            [text],
            labels_embeddings,
            labels,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
            **kwargs
        )[0]

class UniEncoderSpanGLiNER(BaseEncoderGLiNER):
    config_class = UniEncoderSpanConfig 
    model_class = UniEncoderSpanModel
    ort_model_class: Type = UniEncoderSpanORTModel
    data_processor_class = UniEncoderSpanProcessor
    data_collator_class = UniEncoderSpanDataCollator
    decoder_class = SpanDecoder

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        batch = self._build_dummy_batch()
        core = self.model.to("cpu").eval()

        # Required inputs for span uni-encoder
        all_inputs = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["words_mask"],
            batch["text_lengths"],
            batch["span_idx"],
            batch["span_mask"],
        )
        input_names = [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        ]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
            "span_mask": {0: "batch_size", 1: "num_spans"},
            # For UniEncoderSpanORTModel
            "logits": {
                0: "batch_size",
                1: "sequence_length",
                2: "num_spans",
                3: "num_classes",
            },
        }

        class _Wrapper(nn.Module):
            def __init__(self, core_model):
                super().__init__()
                self.core = core_model

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
                logits = out.logits if hasattr(out, "logits") else out[0]
                return logits

        wrapper = _Wrapper(core)
        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            input_names,
            ["logits"],
            dynamic_axes,
            onnx_path,
            opset,
        )

        q_path = self._maybe_quantize_onnx(
            onnx_path, save_dir / quantized_filename, quantize
        )

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }
    
class UniEncoderTokenGLiNER(BaseEncoderGLiNER):
    config_class = UniEncoderTokenConfig 
    model_class = UniEncoderTokenModel
    ort_model_class: Type = UniEncoderTokenORTModel
    data_processor_class = UniEncoderTokenProcessor
    data_collator_class = UniEncoderTokenDataCollator
    decoder_class = TokenDecoder

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        batch = self._build_dummy_batch()
        core = self.model.to("cpu").eval()

        all_inputs = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["words_mask"],
            batch["text_lengths"],
        )
        input_names = ["input_ids", "attention_mask", "words_mask", "text_lengths"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            # For UniEncoderTokenORTModel
            "logits": {
                0: "position",   # as in your original script
                1: "batch_size",
                2: "sequence_length",
                3: "num_classes",
            },
        }

        class _Wrapper(nn.Module):
            def __init__(self, core_model):
                super().__init__()
                self.core = core_model

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
                logits = out.logits if hasattr(out, "logits") else out[0]
                return logits

        wrapper = _Wrapper(core)
        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            input_names,
            ["logits"],
            dynamic_axes,
            onnx_path,
            opset,
        )

        q_path = self._maybe_quantize_onnx(
            onnx_path, save_dir / quantized_filename, quantize
        )

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }
    

class BiEncoderSpanGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderSpanConfig 
    model_class = BiEncoderSpanModel
    ort_model_class: Type = BiEncoderSpanORTModel
    data_processor_class = BiEncoderSpanProcessor
    data_collator_class = BiEncoderSpanDataCollator
    decoder_class = SpanDecoder

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        from_labels_embeddings: bool = False,
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        # Dummy labels so collator/encoder builds label representations
        labels = ["organization", "person", "country"]
        batch = self._build_dummy_batch(labels=labels)
        core = self.model.to("cpu").eval()

        if not from_labels_embeddings:
            all_inputs = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["words_mask"],
                batch["text_lengths"],
                batch["span_idx"],
                batch["span_mask"],
                batch["labels_input_ids"],
                batch["labels_attention_mask"],
            )
            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
                "labels_input_ids",
                "labels_attention_mask",
            ]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
                "span_mask": {0: "batch_size", 1: "num_spans"},
                "labels_input_ids": {0: "num_labels", 1: "label_seq_length"},
                "labels_attention_mask": {0: "num_labels", 1: "label_seq_length"},
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_spans",
                    3: "num_classes",
                },
            }

            class _Wrapper(nn.Module):
                def __init__(self, core_model):
                    super().__init__()
                    self.core = core_model

                def forward(
                    self,
                    input_ids,
                    attention_mask,
                    words_mask,
                    text_lengths,
                    span_idx,
                    span_mask,
                    labels_input_ids,
                    labels_attention_mask,
                ):
                    out = self.core(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        words_mask=words_mask,
                        text_lengths=text_lengths,
                        span_idx=span_idx,
                        span_mask=span_mask,
                        labels_input_ids=labels_input_ids,
                        labels_attention_mask=labels_attention_mask,
                    )
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    return logits

            wrapper = _Wrapper(core)

        else:
            if not hasattr(self, "encode_labels"):
                raise RuntimeError(
                    "from_labels_embeddings=True requires `encode_labels(labels)` "
                    "to be implemented on the bi-encoder model."
                )

            labels_embeds = self.encode_labels(labels).to("cpu")
            all_inputs = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["words_mask"],
                batch["text_lengths"],
                batch["span_idx"],
                batch["span_mask"],
                labels_embeds,
            )
            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "span_idx",
                "span_mask",
                "labels_embeds",  
            ]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
                "span_mask": {0: "batch_size", 1: "num_spans"},
                "labels_embeds": {0: "num_labels", 1: "hidden_size"},
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                    2: "num_spans",
                    3: "num_classes",
                },
            }

            class _Wrapper(nn.Module):
                def __init__(self, core_model):
                    super().__init__()
                    self.core = core_model

                def forward(
                    self,
                    input_ids,
                    attention_mask,
                    words_mask,
                    text_lengths,
                    span_idx,
                    span_mask,
                    labels_embeds,
                ):
                    out = self.core(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        words_mask=words_mask,
                        text_lengths=text_lengths,
                        span_idx=span_idx,
                        span_mask=span_mask,
                        labels_embeds=labels_embeds,
                    )
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    return logits

            wrapper = _Wrapper(core)

        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            input_names,
            ["logits"],
            dynamic_axes,
            onnx_path,
            opset,
        )

        q_path = self._maybe_quantize_onnx(
            onnx_path, save_dir / quantized_filename, quantize
        )

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }
    

class BiEncoderTokenGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderTokenConfig 
    model_class = BiEncoderTokenModel
    ort_model_class: Type = BiEncoderTokenORTModel
    data_processor_class = BiEncoderTokenProcessor
    data_collator_class = BiEncoderTokenDataCollator
    decoder_class = TokenDecoder

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        from_labels_embeddings: bool = False,
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        labels = ["organization", "person", "country"]
        batch = self._build_dummy_batch(labels=labels)
        core = self.model.to("cpu").eval()

        if not from_labels_embeddings:
            all_inputs = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["words_mask"],
                batch["text_lengths"],
                batch["labels_input_ids"],
                batch["labels_attention_mask"],
            )
            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "labels_input_ids",
                "labels_attention_mask",
            ]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "labels_input_ids": {0: "num_labels", 1: "label_seq_length"},
                "labels_attention_mask": {0: "num_labels", 1: "label_seq_length"},
                "logits": {
                    0: "position",
                    1: "batch_size",
                    2: "sequence_length",
                    3: "num_classes",
                },
            }

            class _Wrapper(nn.Module):
                def __init__(self, core_model):
                    super().__init__()
                    self.core = core_model

                def forward(
                    self,
                    input_ids,
                    attention_mask,
                    words_mask,
                    text_lengths,
                    labels_input_ids,
                    labels_attention_mask,
                ):
                    out = self.core(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        words_mask=words_mask,
                        text_lengths=text_lengths,
                        labels_input_ids=labels_input_ids,
                        labels_attention_mask=labels_attention_mask,
                    )
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    return logits

            wrapper = _Wrapper(core)
        else:
            if not hasattr(self, "encode_labels"):
                raise RuntimeError(
                    "from_labels_embeddings=True requires `encode_labels(labels)` "
                    "to be implemented on the bi-encoder model."
                )

            labels_embeds = self.encode_labels(labels).to("cpu")
            all_inputs = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["words_mask"],
                batch["text_lengths"],
                labels_embeds,
            )
            input_names = [
                "input_ids",
                "attention_mask",
                "words_mask",
                "text_lengths",
                "labels_embeds",
            ]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "words_mask": {0: "batch_size", 1: "sequence_length"},
                "text_lengths": {0: "batch_size", 1: "value"},
                "labels_embeds": {0: "num_labels", 1: "hidden_size"},
                "logits": {
                    0: "position",
                    1: "batch_size",
                    2: "sequence_length",
                    3: "num_classes",
                },
            }

            class _Wrapper(nn.Module):
                def __init__(self, core_model):
                    super().__init__()
                    self.core = core_model

                def forward(
                    self,
                    input_ids,
                    attention_mask,
                    words_mask,
                    text_lengths,
                    labels_embeds,
                ):
                    out = self.core(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        words_mask=words_mask,
                        text_lengths=text_lengths,
                        labels_embeds=labels_embeds,
                    )
                    logits = out.logits if hasattr(out, "logits") else out[0]
                    return logits

            wrapper = _Wrapper(core)

        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            input_names,
            ["logits"],
            dynamic_axes,
            onnx_path,
            opset,
        )

        q_path = self._maybe_quantize_onnx(
            onnx_path, save_dir / quantized_filename, quantize
        )

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }
    
class UniEncoderSpanDecoderGLiNER(BaseEncoderGLiNER):
    """
    GLiNER model with span-based encoding and label decoding capabilities.
    Supports generating textual labels for entities.
    """
    config_class = UniEncoderSpanDecoderConfig  # Uses base config with labels_decoder settings
    model_class = UniEncoderSpanDecoderModel
    ort_model_class: Type = None
    data_processor_class = UniEncoderSpanDecoderProcessor
    data_collator_class = UniEncoderSpanDecoderDataCollator
    decoder_class = SpanGenerativeDecoder
    
    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create data processor with decoder tokenizer."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
        
        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)
        
        # Load decoder tokenizer
        decoder_tokenizer = None
        if config.labels_decoder is not None:
            decoder_tokenizer = AutoTokenizer.from_pretrained(
                config.labels_decoder, 
                cache_dir=cache_dir, 
                add_prefix_space=True
            )
            if decoder_tokenizer.pad_token is None:
                decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        
        self.data_processor = self.data_processor_class(
            config, tokenizer, words_splitter, decoder_tokenizer=decoder_tokenizer
        )
        return self.data_processor
    
    def set_labels_trie(self, labels: List[str]):
        """
        Initialize the labels trie for constrained generation.
        
        Args:
            labels (List[str]): Labels that will be used for constrained generation.
            
        Returns:
            LabelsTrie: Trie structure for constrained beam search.
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
        """
        Generate textual class labels for each entity span.
        
        Args:
            model_output: Model output containing decoder_embedding and decoder_embedding_mask
            **gen_kwargs: Generation parameters (max_new_tokens, temperature, etc.)
            
        Returns:
            List[str]: Generated label strings
        """
        dec_embeds = model_output.decoder_embedding
        if dec_embeds is None:
            return []
        
        dec_mask = model_output.decoder_embedding_mask
        
        gen_ids = self.model.generate_labels(
            dec_embeds, dec_mask,
            max_new_tokens=gen_kwargs.pop("max_new_tokens", 15),
            eos_token_id=self.data_processor.decoder_tokenizer.eos_token_id,
            pad_token_id=self.data_processor.decoder_tokenizer.pad_token_id,
            do_sample=gen_kwargs.pop("do_sample", True),
            temperature=gen_kwargs.pop("temperature", 0.01),
            num_return_sequences=gen_kwargs.pop("num_return_sequences", 1),
            **gen_kwargs
        )
        
        gen_texts = self.data_processor.decoder_tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True
        )
        return gen_texts
    
    @torch.no_grad()
    def inference(
        self,
        texts,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        batch_size=8,
        gen_constraints=None,
        num_gen_sequences=1,
        packing_config: Optional[InferencePackingConfig] = None,
        **gen_kwargs,
    ):
        """
        Predict entities with optional label generation.
        
        Args:
            texts: Input texts
            labels: Entity type labels
            flat_ner: Whether to use flat NER
            threshold: Confidence threshold
            multi_label: Allow multiple labels per span
            batch_size: Batch size for processing
            gen_constraints: Labels to constrain generation
            num_gen_sequences: Number of label sequences to generate per span
            packing_config: Inference packing configuration
            **gen_kwargs: Additional generation parameters
            
        Returns:
            List of entity predictions with optional generated labels
        """
        self.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        entity_types = list(dict.fromkeys(labels))

        tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_inputs(texts)
        input_x = self.prepare_base_input(tokens)
                
        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        def collate_fn(batch, entity_types=entity_types):
            batch_out = collator(batch, entity_types=entity_types)
            return batch_out

        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        active_packing = packing_config if packing_config is not None else self._inference_packing_config
        
        outputs = []
        for batch in data_loader:
            if not self.onnx_model:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            model_inputs = batch.copy() if active_packing is None else {**batch, "packing_config": active_packing}
            model_output = self.model(**model_inputs, threshold=threshold)
            
            model_logits = model_output.logits
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)
            
            # Generate labels if decoder is available
            gen_labels = None
            if self.config.labels_decoder is not None:
                labels_trie = self.set_labels_trie(gen_constraints) if gen_constraints else None
                gen_labels = self.generate_labels(
                    model_output, 
                    labels_trie=labels_trie,
                    num_return_sequences=num_gen_sequences,
                    **gen_kwargs
                )
            
            decoded = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_logits,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
                gen_labels=gen_labels,
                sel_idx=model_output.decoder_span_idx,
                num_gen_sequences=num_gen_sequences
            )
            outputs.extend(decoded)
        
        # Convert to entity format
        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
            entities = []
            
            for start_token_idx, end_token_idx, ent_type, gen_ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                
                ent_details = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": texts[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                }
                
                if gen_ent_type is not None:
                    ent_details['generated_labels'] = gen_ent_type
                
                entities.append(ent_details)
            
            all_entities.append(entities)
        
        return all_entities

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        raise NotImplementedError(
            "ONNX export is not supported for encoder-decoder GLiNER models "
            "(UniEncoderSpanDecoderGLiNER) because of the generative decoder head. "
            "Export the encoder-only variant or add a dedicated export pipeline."
        )
    
class UniEncoderSpanRelexGLiNER(BaseEncoderGLiNER):
    """
    GLiNER model for both entity recognition and relation extraction.
    Performs joint entity and relation prediction.
    """
    config_class = UniEncoderSpanRelexConfig
    model_class = UniEncoderSpanRelexModel
    ort_model_class: Type = UniEncoderSpanRelexORTModel
    data_processor_class = RelationExtractionSpanProcessor
    data_collator_class = RelationExtractionSpanDataCollator
    decoder_class = SpanDecoder
    
    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        """Create relation extraction data processor."""
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)
        
        if words_splitter is None:
            words_splitter = WordsSplitter(config.words_splitter_type)
        
        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter)
        return self.data_processor
    
    @torch.no_grad()
    def inference(
        self,
        texts,
        labels,
        flat_ner=True,
        threshold=0.5,
        multi_label=False,
        batch_size=8,
        packing_config: Optional[InferencePackingConfig] = None,
        return_relations=True,
    ):
        """
        Predict entities and relations.
        
        Args:
            texts: Input texts
            labels: Entity type labels
            flat_ner: Whether to use flat NER
            threshold: Confidence threshold
            multi_label: Allow multiple labels per span
            batch_size: Batch size
            packing_config: Inference packing configuration
            return_relations: Whether to return relation predictions
            
        Returns:
            Tuple of (entities, relations) if return_relations=True, else just entities
        """
        self.eval()
        
        if isinstance(texts, str):
            texts = [texts]
        
        tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_inputs(texts)
        input_x = self.prepare_base_input(tokens)
        
        collator = self.data_collator_class(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            return_rel_id_to_classes=True,
            prepare_labels=False,
        )
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collator
        )
        
        active_packing = packing_config if packing_config is not None else self._inference_packing_config
        
        all_entity_outputs = []
        all_relation_outputs = []
        
        for batch in data_loader:
            if not self.onnx_model:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            model_inputs = batch.copy() if active_packing is None else {**batch, "packing_config": active_packing}
            model_output = self.model(**model_inputs, threshold=threshold)
            
            # Decode entities
            model_logits = model_output.logits
            if not isinstance(model_logits, torch.Tensor):
                model_logits = torch.from_numpy(model_logits)
            
            decoded_entities = self.decoder.decode(
                batch["tokens"],
                batch["id_to_classes"],
                model_logits,
                flat_ner=flat_ner,
                threshold=threshold,
                multi_label=multi_label,
            )
            all_entity_outputs.extend(decoded_entities)
            
            # Store relations if available
            if return_relations and hasattr(model_output, 'rel_logits') and model_output.rel_logits is not None:
                all_relation_outputs.append({
                    'rel_idx': model_output.rel_idx,
                    'rel_logits': model_output.rel_logits,
                    'rel_mask': model_output.rel_mask,
                })
        
        # Convert entities to standard format
        all_entities = []
        for i, output in enumerate(all_entity_outputs):
            start_token_idx_to_text_idx = all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = all_end_token_idx_to_text_idx[i]
            entities = []
            
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                
                entities.append({
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": texts[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                })
            
            all_entities.append(entities)
        
        if return_relations:
            # Process relations
            all_relations = self._process_relations(
                all_relation_outputs, all_entities, threshold
            )
            return all_entities, all_relations
        
        return all_entities
    
    def _process_relations(self, relation_outputs, all_entities, threshold=0.5):
        """Process relation predictions into readable format."""
        all_relations = []
        
        for rel_output in relation_outputs:
            if rel_output is None:
                all_relations.append([])
                continue
            
            rel_idx = rel_output['rel_idx']  # (B, P, 2)
            rel_logits = rel_output['rel_logits']  # (B, P, C)
            rel_mask = rel_output['rel_mask']  # (B, P)
            
            batch_relations = []
            for b in range(rel_idx.size(0)):
                relations = []
                for p in range(rel_idx.size(1)):
                    if not rel_mask[b, p]:
                        continue
                    
                    head_idx = rel_idx[b, p, 0].item()
                    tail_idx = rel_idx[b, p, 1].item()
                    
                    # Get relation type with highest score
                    scores = torch.sigmoid(rel_logits[b, p])
                    max_score, max_idx = scores.max(dim=0)
                    
                    if max_score >= threshold:
                        relations.append({
                            'head': head_idx,
                            'tail': tail_idx,
                            'relation': max_idx.item(),
                            'score': max_score.item(),
                        })
                
                batch_relations.append(relations)
            
            all_relations.extend(batch_relations)
        
        return all_relations

    def export_to_onnx(
        self,
        save_dir: Union[str, Path],
        onnx_filename: str = "model.onnx",
        quantized_filename: str = "model_quantized.onnx",
        quantize: bool = False,
        opset: int = 19,
    ) -> Dict[str, Optional[str]]:
        self._check_onnx_export_preconditions()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = save_dir / onnx_filename

        # Use dummy labels; collator will construct spans + relation candidates
        batch = self._build_dummy_batch(labels=["head", "tail"])
        core = self.model.to("cpu").eval()

        all_inputs = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["words_mask"],
            batch["text_lengths"],
            batch["span_idx"],
            batch["span_mask"],
        )
        input_names = [
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        ]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "words_mask": {0: "batch_size", 1: "sequence_length"},
            "text_lengths": {0: "batch_size", 1: "value"},
            "span_idx": {0: "batch_size", 1: "num_spans", 2: "idx"},
            "span_mask": {0: "batch_size", 1: "num_spans"},
            # For UniEncoderSpanRelexORTModel
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
        }

        class _Wrapper(nn.Module):
            def __init__(self, core_model):
                super().__init__()
                self.core = core_model

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
                # GLiNERRelexOutput expected
                logits = out.logits
                rel_idx = out.rel_idx
                rel_logits = out.rel_logits
                rel_mask = out.rel_mask
                return logits, rel_idx, rel_logits, rel_mask

        wrapper = _Wrapper(core)
        self._run_torch_onnx_export(
            wrapper,
            all_inputs,
            input_names,
            ["logits", "rel_idx", "rel_logits", "rel_mask"],
            dynamic_axes,
            onnx_path,
            opset,
        )

        # Quantization for multi-output models is still fine (weights only)
        q_path = self._maybe_quantize_onnx(
            onnx_path, save_dir / quantized_filename, quantize
        )

        return {
            "onnx_path": str(onnx_path),
            "quantized_path": str(q_path) if q_path is not None else None,
        }


class GLiNER(nn.Module, PyTorchModelHubMixin):
    """
    Meta GLiNER class that automatically instantiates the appropriate GLiNER variant.
    
    This class provides a unified interface for all GLiNER models, automatically switching to 
    specialized model types based on the model configuration. It supports various NER architectures
    including uni-encoder, bi-encoder, decoder-based, and relation extraction models.
    
    The class automatically detects the model type based on:
    - span_mode: Token-level vs span-level
    - labels_encoder: Uni-encoder vs bi-encoder
    - labels_decoder: Standard vs decoder-based
    - relations_layer: NER-only vs joint entity-relation extraction
    
    Attributes:
        model: The loaded GLiNER model instance (automatically typed)
        config: Model configuration
        data_processor: Data processor for the model
        decoder: Decoder for predictions
        
    Examples:
        Load a pretrained uni-encoder span model:
        >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        
        Load a bi-encoder model:
        >>> model = GLiNER.from_pretrained("urchade/gliner_bi-small-v1.0")
        
        Load from local configuration:
        >>> config = GLiNERConfig.from_pretrained("config.json")
        >>> model = GLiNER.from_config(config)
        
        Initialize from scratch:
        >>> config = GLiNERConfig(model_name="microsoft/deberta-v3-small")
        >>> model = GLiNER(config)
    """
    
    def __init__(
        self, 
        config: Union[str, Path, GLiNERConfig],
        **kwargs
    ):
        """
        Initialize a GLiNER model with automatic type detection.
        
        This constructor determines the appropriate GLiNER variant based on the configuration
        and replaces itself with an instance of that variant.
        
        Args:
            config: Model configuration (GLiNERConfig object, path to config file, or dict)
            **kwargs: Additional arguments passed to the specific GLiNER variant
            
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
                with open(config_path, "r") as f:
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
        """
        Determine the appropriate GLiNER class based on configuration.
        
        Args:
            config: GLiNER configuration object
            
        Returns:
            The appropriate GLiNER class
        """
        is_token_level = config.span_mode == "token_level"
        has_labels_encoder = config.labels_encoder is not None
        has_labels_decoder = config.labels_decoder is not None
        has_relations = config.relations_layer is not None
        
        # Priority order: relations > decoder > bi-encoder > token vs span
        
        if has_relations:
            return UniEncoderSpanRelexGLiNER
        
        if has_labels_decoder:
            if has_labels_encoder:
                warnings.warn(
                    "labels_encoder and labels_decoder are both set. "
                    "Using decoder model (labels_encoder will be ignored)."
                )
            return UniEncoderSpanDecoderGLiNER
        
        if has_labels_encoder:
            if is_token_level:
                return BiEncoderTokenGLiNER
            else:
                return BiEncoderSpanGLiNER
        
        # Default: uni-encoder
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
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Union[str, bool, None] = None,
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = None,
        resize_token_embeddings: Optional[bool] = True,
        compile_torch_model: Optional[bool] = False,
        load_onnx_model: Optional[bool] = False,
        onnx_model_file: Optional[str] = "model.onnx",
        # Config overrides
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        _attn_implementation: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Load a pretrained GLiNER model with automatic type detection.
        
        This method loads the configuration, determines the appropriate GLiNER variant,
        and delegates to that variant's from_pretrained method.
        
        Args:
            model_id: Model identifier or local path
            revision: Model revision
            cache_dir: Cache directory
            force_download: Force redownload
            proxies: Proxy configuration
            resume_download: Resume interrupted downloads
            local_files_only: Only use local files
            token: HF token for private repos
            map_location: Device to map model to
            strict: Enforce strict state_dict loading
            load_tokenizer: Whether to load tokenizer
            resize_token_embeddings: Whether to resize embeddings
            compile_torch_model: Whether to compile with torch.compile
            max_length: Override max_length in config
            max_width: Override max_width in config
            post_fusion_schema: Override post_fusion_schema in config
            _attn_implementation: Override attention implementation
            **model_kwargs: Additional model initialization arguments
            
        Returns:
            Appropriate GLiNER model instance
            
        Examples:
            >>> model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            >>> model = GLiNER.from_pretrained("urchade/gliner_bi-small-v1.0")
            >>> model = GLiNER.from_pretrained("path/to/local/model")
        """
        model_dir = Path(model_id)
        if not model_dir.exists():
            model_dir = Path(snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            ))
        
        # Load config to determine model type
        config_file = model_dir / "gliner_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config file found in {model_dir}")
        
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        config_dict.pop("model_type", None)

        config = GLiNERConfig(**config_dict)
        
        # Determine the appropriate class
        gliner_class = cls._get_gliner_class(config)
        
        print(f"Loading the following GLiNER type: {gliner_class}...")
        # Delegate to the specific class's from_pretrained method
        return gliner_class.from_pretrained(
            model_id=model_id,
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
        config: Union[GLiNERConfig, str, Path, Dict],
        **kwargs
    ):
        """
        Create a GLiNER model from configuration.
        
        Args:
            config: Model configuration (GLiNERConfig, path, or dict)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Appropriate GLiNER model instance
            
        Examples:
            >>> config = GLiNERConfig(model_name="microsoft/deberta-v3-small")
            >>> model = GLiNER.from_config(config)
            
            >>> model = GLiNER.from_config("path/to/config.json")
        """
        # Load config if needed
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                config = GLiNERConfig(**config_dict)
            else:
                raise FileNotFoundError(f"Config file not found: {config}")
        elif isinstance(config, dict):
            config = GLiNERConfig(**config)
        
        # Determine the appropriate class
        gliner_class = cls._get_gliner_class(config)
        
        # Create instance
        return gliner_class(config, **kwargs)
    
    @property
    def model_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Map configuration patterns to their corresponding GLiNER classes.
        
        Returns:
            Dictionary mapping model types to their classes and descriptions
        """
        return {
            "uni_encoder_span": {
                "class": UniEncoderSpanGLiNER,
                "description": "Standard span-based NER with single encoder",
                "config": {"span_mode": "span_level", "labels_encoder": None, "labels_decoder": None, "relations_layer": None},
            },
            "uni_encoder_token": {
                "class": UniEncoderTokenGLiNER,
                "description": "Token-level NER with single encoder",
                "config": {"span_mode": "token_level", "labels_encoder": None, "labels_decoder": None, "relations_layer": None},
            },
            "bi_encoder_span": {
                "class": BiEncoderSpanGLiNER,
                "description": "Span-based NER with separate text and label encoders",
                "config": {"span_mode": "span_level", "labels_encoder": "required", "labels_decoder": None, "relations_layer": None},
            },
            "bi_encoder_token": {
                "class": BiEncoderTokenGLiNER,
                "description": "Token-level NER with separate text and label encoders",
                "config": {"span_mode": "token_level", "labels_encoder": "required", "labels_decoder": None, "relations_layer": None},
            },
            "span_decoder": {
                "class": UniEncoderSpanDecoderGLiNER,
                "description": "Span-based NER with label generation decoder",
                "config": {"span_mode": "span_level", "labels_decoder": "required", "relations_layer": None},
            },
            "span_relex": {
                "class": UniEncoderSpanRelexGLiNER,
                "description": "Joint entity and relation extraction with single encoder",
                "config": {"span_mode": "span_level", "labels_encoder": None, "relations_layer": "required"},
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
            "UniEncoderSpanGLiNER": "uni_encoder_span",
            "UniEncoderTokenGLiNER": "uni_encoder_token",
            "BiEncoderSpanGLiNER": "bi_encoder_span",
            "BiEncoderTokenGLiNER": "bi_encoder_token",
            "UniEncoderSpanDecoderGLiNER": "span_decoder",
            "UniEncoderSpanRelexGLiNER": "span_relex",
            "BiEncoderSpanRelexGLiNER": "bi_span_relex",
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
