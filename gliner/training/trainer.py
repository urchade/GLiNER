"""Custom Trainer implementation with enhanced loss functions and optimizer configuration.

This module extends the Hugging Face Transformers Trainer class to support
custom loss functions (focal loss, label smoothing), flexible learning rates
for different parameter groups, and robust error handling during training.
"""

import os
import math
import inspect
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path
from dataclasses import field, dataclass

import torch
import transformers
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_utils import set_seed


def _get_trainer_imports():
    from transformers.trainer import get_parameter_names, is_sagemaker_mp_enabled  # noqa: PLC0415

    return get_parameter_names, is_sagemaker_mp_enabled


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.getLogger(__name__)


def seed_worker(_):
    """Set worker seed during DataLoader initialization.

    Helper function to ensure reproducibility by seeding each DataLoader worker
    process with a unique but deterministic seed based on PyTorch's initial seed.

    Args:
        _: Worker ID (unused, but required by DataLoader worker_init_fn signature).
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments with custom loss and optimization parameters.

    Extends the standard Hugging Face TrainingArguments with additional parameters
    for focal loss, label smoothing, differential learning rates, and custom
    negative sampling strategies.

    Attributes:
        cache_dir: Directory to cache downloaded models and datasets.
        optim: Optimizer to use. Defaults to "adamw_torch".
        others_lr: Optional separate learning rate for non-encoder parameters
            (e.g., classification heads). If None, uses the main learning rate.
        others_weight_decay: Weight decay for non-encoder parameters when
            using others_lr. Defaults to 0.0.
        focal_loss_alpha: Alpha parameter for focal loss. Values < 0 disable
            focal loss weighting. Defaults to -1.
        focal_loss_gamma: Gamma (focusing parameter) for focal loss. Higher values
            increase focus on hard examples. Defaults to 0.
        focal_loss_prob_margin: Probability margin for focal loss computation.
            Defaults to 0.
        label_smoothing: Label smoothing factor. 0.0 means no smoothing.
            Defaults to 0.
        loss_reduction: Reduction method for loss ('sum', 'mean', or 'none').
            Defaults to 'sum'.
        negatives: Ratio of negative samples to use. Defaults to 1.0.
        masking: Masking strategy for training ('global' or other strategies).
            Defaults to 'global'.
        warmup_ratio: Fraction of training steps used for warmup when
            warmup_steps is zero. Retained for compatibility with Transformers v4.
    """

    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: float | None = None
    others_weight_decay: float | None = 0.0
    focal_loss_alpha: float | None = -1
    focal_loss_gamma: float | None = 0
    rel_focal_loss_alpha: float | None = None
    rel_focal_loss_gamma: float | None = None
    focal_loss_prob_margin: float | None = 0
    label_smoothing: float | None = 0
    loss_reduction: str | None = "sum"
    negatives: float | None = 1.0
    masking: str | None = "global"
    loss_type: str | None = "focal"
    use_span_width_weight: bool | None = False
    dice_gamma: float | None = 1.0
    warmup_ratio: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must lie in range [0, 1]")
        super().__post_init__()

    def get_warmup_steps(self, num_training_steps: int):
        # New Transformers versions accept ratios through warmup_steps instead.
        # Keep the legacy ratio separate so 1.0 still means all training steps,
        # and epoch-based training uses the actual step count from Trainer.
        if self.warmup_steps == 0:
            return math.ceil(num_training_steps * self.warmup_ratio)
        return super().get_warmup_steps(num_training_steps)


class Trainer(transformers.Trainer):
    """Transformers v4/v5 compatible custom Trainer.

    - v5-safe method signatures (num_items_in_batch)
    - no hard dependency on self.use_apex
    - skips only OOM by default (other exceptions are raised so you don't silently get 0 loss)
    """

    def _load_gliner_checkpoint(self, checkpoint, model) -> bool:
        """Restore native GLiNER weights in place; defer other formats to Transformers."""
        from gliner.model import BaseGLiNER  # noqa: PLC0415

        _, is_sagemaker_mp_enabled = _get_trainer_imports()
        if self.is_deepspeed_enabled or self.is_fsdp_enabled or is_sagemaker_mp_enabled():
            return False

        model = self.accelerator.unwrap_model(model)
        model = getattr(model, "_orig_mod", model)
        checkpoint = Path(checkpoint)
        if not isinstance(model, BaseGLiNER) or not (checkpoint / "gliner_config.json").is_file():
            return False

        # Match from_pretrained's preference for safetensors. Its loader also
        # restores aliases omitted from safetensors files with shared weights.
        model_file = checkpoint / "model.safetensors"
        if not model_file.is_file():
            model_file = checkpoint / "pytorch_model.bin"
        if not model_file.is_file():
            return False

        logger.info("Loading GLiNER model from %s", checkpoint)
        state_dict = model._load_state_dict(model_file, map_location="cpu")
        # save_pretrained serializes the inner model, without the wrapper's
        # `model.` prefix. Copy into existing parameters to preserve optimizer
        # references when resuming training.
        inner_model = getattr(model.model, "_orig_mod", model.model)
        load_result = inner_model.load_state_dict(state_dict, strict=False)
        del state_dict
        self._issue_warnings_after_load(load_result)
        return True

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        if not self._load_gliner_checkpoint(resume_from_checkpoint, target):
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_best_model(self):
        if not self._load_gliner_checkpoint(self.state.best_model_checkpoint, self.model):
            return super()._load_best_model()

    def _save(self, output_dir: str | None = None, state_dict=None):
        # called by HF during checkpoint saves
        if not self.args.should_save:
            return

        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.accelerator.unwrap_model(self.model)

        # Prefer safetensors if TrainingArguments says so
        safe = bool(getattr(self.args, "save_safetensors", False))

        sp = getattr(model_to_save, "save_pretrained", None)
        if sp is None:
            # last-resort fallback: behave like HF (weights only)
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            return

        sp_sig = inspect.signature(sp).parameters
        kwargs = {}
        if "safe_serialization" in sp_sig:
            kwargs["safe_serialization"] = safe

        if state_dict is not None and "state_dict" in sp_sig:
            kwargs["state_dict"] = state_dict

        model_to_save.save_pretrained(output_dir, **kwargs)

        proc = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(output_dir)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        # make final save consistent with checkpoint saving
        self._save(output_dir)

    @property
    def use_apex(self) -> bool:
        return bool(getattr(self, "_use_apex", False))

    @use_apex.setter
    def use_apex(self, value: bool) -> None:
        self._use_apex = bool(value)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        # Prepare inputs are done in training_step / prediction_step
        rel_alpha = (
            self.args.rel_focal_loss_alpha if self.args.rel_focal_loss_alpha is not None else self.args.focal_loss_alpha
        )
        rel_gamma = (
            self.args.rel_focal_loss_gamma if self.args.rel_focal_loss_gamma is not None else self.args.focal_loss_gamma
        )
        outputs = model(
            alpha=self.args.focal_loss_alpha,
            gamma=self.args.focal_loss_gamma,
            rel_alpha=rel_alpha,
            rel_gamma=rel_gamma,
            prob_margin=self.args.focal_loss_prob_margin,
            label_smoothing=self.args.label_smoothing,
            reduction=self.args.loss_reduction,
            negatives=self.args.negatives,
            masking=self.args.masking,
            loss_type=self.args.loss_type,
            use_span_width_weight=self.args.use_span_width_weight,
            dice_gamma=self.args.dice_gamma,
            **inputs,
        )

        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Guardrail: if labels are missing, fail loudly (otherwise you end up with loss=None -> silent 0)
        if "labels" not in inputs:
            raise KeyError(f"Batch has no 'labels'. Keys: {list(inputs.keys())}")

        try:
            _, is_sagemaker_mp_enabled = _get_trainer_imports()
            if is_sagemaker_mp_enabled():
                from transformers.trainer_pt_utils import smp_forward_backward  # noqa: PLC0415

                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            if loss is None:
                raise RuntimeError("Model returned loss=None (check labels / remove_unused_columns / forward).")

            # Average on multi-gpu
            if self.args.n_gpu > 1:
                loss = loss.mean()

            # Match upstream Trainer behavior: scale loss for grad accumulation before backward
            if self.args.gradient_accumulation_steps > 1 and self.deepspeed is None:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)

            return loss.detach()

        except torch.cuda.OutOfMemoryError as e:
            logger.warning("Skipping batch due to CUDA OOM: %s", e)
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return torch.zeros((), device=self.args.device)

        except RuntimeError as e:
            # Some OOMs come as RuntimeError("CUDA out of memory...")
            if "out of memory" in str(e).lower():
                logger.warning("Skipping batch due to OOM RuntimeError: %s", e)
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros((), device=self.args.device)
            # Anything else: raise, so you don't silently train with zeros again
            raise

    def create_optimizer(self):
        get_parameter_names, is_sagemaker_mp_enabled = _get_trainer_imports()
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        if self.args.others_lr is not None:
            encoder_parameters = [name for name, _ in opt_model.named_parameters() if "token_rep_layer" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.others_weight_decay,
                    "lr": self.args.others_lr,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.others_lr,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # Works across v4/v5
        if hasattr(transformers.Trainer, "get_optimizer_cls_and_kwargs"):
            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        else:
            # very old fallback
            optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: List[str] | None = None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        model.eval()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            logits = getattr(outputs, "logits", None)
            labels = inputs.get("labels", None)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)

        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
