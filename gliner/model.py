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

from .config import (BaseGLiNERConfig, 
                     UniEncoderSpanConfig, 
                     UniEncoderTokenConfig,
                     BiEncoderSpanConfig,
                     BiEncoderTokenConfig,
                     GLiNERConfig)
from .data_processing import (BaseProcessor, UniEncoderSpanProcessor, UniEncoderTokenProcessor, 
                                                BiEncoderSpanProcessor, BiEncoderTokenProcessor)
from .data_processing.collator import DataCollator, DataCollatorWithPadding
from .data_processing.tokenizer import WordsSplitter
from .decoding import SpanDecoder, TokenDecoder
from .decoding.trie import LabelsTrie
from .evaluation import BaseNEREvaluator
from .training import TrainingArguments, Trainer
from .modeling.base import (BaseModel, 
                            UniEncoderSpanModel, 
                            UniEncoderTokenModel,
                            BiEncoderSpanModel,
                            BiEncoderTokenModel)
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


class BaseEncoderGLiNER(BaseGLiNER):
    def _create_model(self, config, backbone_from_pretrained, cache_dir, **kwargs):
         self.model = self.model_class(config, backbone_from_pretrained, cache_dir=cache_dir, **kwargs)
    
    def _create_data_processor(self, config, cache_dir, tokenizer=None, words_splitter=None, **kwargs):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=cache_dir)

        self.data_processor = self.data_processor_class(config, tokenizer, words_splitter)

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

        tokens, all_start_token_idx_to_text_idx, all_end_token_idx_to_text_idx = self.prepare_texts(texts)
        
        input_x = self.prepare_base_input(tokens)

        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=labels,
        )
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collator
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
            "Please use GLiNER.run instead.",
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
        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=entity_types,
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

    def resize_embeddings(self):
        warnings.warn("Resizing embeddings is not supported for bi-encoder models.")

    @torch.no_grad()
    def batch_predict_with_embeds(
        self,
        texts,
        labels_embeddings,
        labels,
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
            labels (List[str]): A list of labels to predict.
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

        collator = DataCollator(
            self.config,
            data_processor=self.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
            entity_types=labels,
        )
        data_loader = torch.utils.data.DataLoader(
            input_x, batch_size=batch_size, shuffle=False, collate_fn=collator
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
    data_processor_class = UniEncoderSpanProcessor
    decoder_class = SpanDecoder

class UniEncoderTokenGLiNER(BaseEncoderGLiNER):
    config_class = UniEncoderTokenConfig 
    model_class = UniEncoderTokenModel
    data_processor_class = UniEncoderTokenProcessor
    decoder_class = TokenDecoder


class BiEncoderSpanGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderSpanConfig 
    model_class = BiEncoderSpanModel
    data_processor_class = BiEncoderSpanProcessor
    decoder_class = SpanDecoder

class BiEncoderTokenGLiNER(BaseBiEncoderGLiNER):
    config_class = BiEncoderTokenConfig 
    model_class = BiEncoderTokenModel
    data_processor_class = BiEncoderTokenProcessor
    decoder_class = TokenDecoder