import json
import os
import re
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union, Type
from abc import ABC, abstractmethod

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

from .config import BaseGLiNERConfig, GLiNERConfig
from .data_processing import BaseProcessor
from .data_processing.collator import DataCollator, DataCollatorWithPadding
from .data_processing.tokenizer import WordsSplitter
from .decoding import SpanDecoder, TokenDecoder
from .decoding.trie import LabelsTrie
from .evaluation import Evaluator
from .training import TrainingArguments, Trainer
from .modeling.base import BaseModel
from .infer_packing import InferencePackingConfig
from .onnx.model import BaseORTModel, SpanORTModel, TokenORTModel
from .utils import is_module_available

if is_module_available("onnxruntime"):
    import onnxruntime as ort
    ONNX_AVAILABLE = True
else:
    ONNX_AVAILABLE = False

class BaseGLiNER(ABC, nn.Module, PyTorchModelHubMixin):
    config_class: Type = None 
    model_class: Type = None
    data_processor_class: Type = None
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
            self.data_processor = self._create_data_processor(config, tokenizer, cache_dir, **kwargs)
    
        if isinstance(self.model, BaseORTModel):
            self.onnx_model = True
        else:
            self.onnx_model = False

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
    def prepare_model_inputs(self):
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
    def load_from_config(gliner_config: GLiNERConfig):
        pass

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
        return instance

    def _create_data_collator(self, use_new_schema: bool = False):
        """
        Create data collator. Override in child classes if needed.
        
        Args:
            use_new_schema: Whether to use new data schema
            
        Returns:
            Data collator instance
        """
        if use_new_schema:
            return DataCollatorWithPadding(self.config)
        else:
            return DataCollator(
                self.config, 
                data_processor=self.data_processor, 
                prepare_labels=True
            )
    
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
            print(f"Warning: Component '{component_name}' not found or not freezable")
    
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
            print(f"Warning: Component '{component_name}' not found")
    

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
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        training_args: Optional[TrainingArguments] = None,
        freeze_components: Optional[List[str]] = None,
        compile_model: bool = False,
        use_new_data_schema: bool = False,
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
        data_collator = self._create_data_collator(use_new_data_schema)
        
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
