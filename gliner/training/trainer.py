"""Custom Trainer implementation with enhanced loss functions and optimizer configuration.

This module extends the Hugging Face Transformers Trainer class to support
custom loss functions (focal loss, label smoothing), flexible learning rates
for different parameter groups, and robust error handling during training.
"""

import logging
from typing import Any, Dict, List, Tuple, Union, Optional
from dataclasses import field, dataclass

import torch
import transformers
from torch import nn
from transformers.trainer import (
    get_parameter_names,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_utils import set_seed

if transformers.utils.is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
from torch.utils.data import Dataset, DataLoader

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
    """

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: Optional[float] = None
    others_weight_decay: Optional[float] = 0.0
    focal_loss_alpha: Optional[float] = -1
    focal_loss_gamma: Optional[float] = 0
    focal_loss_prob_margin: Optional[float] = 0
    label_smoothing: Optional[float] = 0
    loss_reduction: Optional[str] = "sum"
    negatives: Optional[float] = 1.0
    masking: Optional[str] = "global"


class Trainer(transformers.Trainer):
    """Custom Trainer with enhanced loss functions and error handling.

    Extends the Hugging Face Trainer to support:
    - Custom loss functions (focal loss, label smoothing)
    - Differential learning rates for encoder vs. other parameters
    - Robust error handling with automatic recovery from failed batches
    - Custom negative sampling and masking strategies
    - Persistent worker support for data loading

    The trainer automatically handles CUDA out-of-memory errors and other
    exceptions during training by skipping problematic batches and continuing.
    """

    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Perform a training step on a batch of inputs.

        Executes forward pass, loss computation, and backward pass for a single
        training batch. Includes automatic error handling to skip problematic
        batches without crashing the training run.

        Args:
            model: The model to train.
            inputs: Dictionary of input tensors and targets for the model.
                The dictionary will be unpacked before being fed to the model.
                Most models expect targets under the 'labels' key.
            *args: Additional positional arguments (unused, for compatibility).
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            Training loss tensor for this batch, scaled by gradient accumulation
            steps. Returns a zero tensor with requires_grad=True if an error occurs.

        Note:
            If an exception occurs during the training step, the method prints
            the error, zeros gradients, clears CUDA cache, and returns a zero
            loss to allow training to continue.
        """
        model.train()
        try:
            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            del inputs
            torch.cuda.empty_cache()

            kwargs = {}

            if self.args.n_gpu > 1:
                loss = loss.mean()  # Average on multi-gpu training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss, **kwargs)

            return loss.detach() / self.args.gradient_accumulation_steps

        except Exception as e:
            logger.info("Skipping iteration due to error: %s", e)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # Safely get device for DataParallel or normal model
            _model = getattr(model, "module", model)
            device = next(_model.parameters()).device
            return torch.tensor(0.0, requires_grad=True, device=device)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save the trained model to a directory.

        Args:
            output_dir: Directory path where the model should be saved.
                If None, uses the default output directory from training arguments.
            _internal_call: Whether this is an internal call from the Trainer.
                Used for compatibility with the parent class.
        """
        self.model.save_pretrained(output_dir)

    def compute_loss(self, model, inputs):
        """Compute loss using custom loss functions.

        Performs forward pass with custom loss parameters including focal loss,
        label smoothing, and negative sampling configurations from training arguments.

        Args:
            model: The model to compute loss for.
            inputs: Dictionary of input tensors including features and labels.

        Returns:
            Computed loss tensor.

        Note:
            The loss function parameters (alpha, gamma, label_smoothing, etc.)
            are passed to the model's forward method, so the model must support
            these keyword arguments.
        """
        # Forward pass
        outputs = model(
            alpha=self.args.focal_loss_alpha,
            gamma=self.args.focal_loss_gamma,
            prob_margin=self.args.focal_loss_prob_margin,
            label_smoothing=self.args.label_smoothing,
            reduction=self.args.loss_reduction,
            negatives=self.args.negatives,
            masking=self.args.masking,
            **inputs,
        )
        loss = outputs.loss
        return loss

    def create_optimizer(self):
        """Create and configure the optimizer with parameter groups.

        Sets up the optimizer with support for:
        - Separate learning rates for encoder and non-encoder parameters
        - Weight decay only for non-bias and non-LayerNorm parameters
        - Custom weight decay values for different parameter groups

        Returns:
            Configured optimizer instance.

        Note:
            If self.args.others_lr is set, creates four parameter groups:
            1. Non-encoder parameters with weight decay
            2. Non-encoder parameters without weight decay
            3. Encoder parameters with weight decay
            4. Encoder parameters without weight decay

            Otherwise, creates two standard parameter groups with and without
            weight decay.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
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
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Perform an evaluation step on the model using inputs.

        Executes a single forward pass for evaluation without computing gradients.

        Args:
            model: The model to evaluate.
            inputs: Dictionary of input tensors and targets for the model.
                The dictionary will be unpacked before being fed to the model.
                Most models expect targets under the 'labels' key.
            prediction_loss_only: If True, only returns the loss and ignores
                logits and labels.
            ignore_keys: Optional list of keys in the model output dictionary
                that should be ignored when gathering predictions. Currently unused.

        Returns:
            A tuple of (loss, logits, labels):
            - loss: Loss tensor if computed, None otherwise
            - logits: Model predictions if prediction_loss_only is False, None otherwise
            - labels: Ground truth labels if prediction_loss_only is False, None otherwise
        """
        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            labels = inputs["labels"]
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)

    def get_train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Constructs a DataLoader with appropriate sampler, collation function,
        and worker configuration for the training dataset. Includes seeded
        worker initialization for reproducibility.

        Returns:
            Configured and accelerator-prepared training DataLoader.

        Raises:
            ValueError: If train_dataset is None.

        Note:
            For IterableDataset, sampler and drop_last are not set.
            For regular datasets, uses the sampler from _get_train_sampler()
            and applies worker seeding via seed_worker function.
        """
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

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """Create and return the evaluation DataLoader.

        Constructs a DataLoader for evaluation with support for persistent workers
        and multiple evaluation datasets. Caches DataLoaders when persistent workers
        are enabled to avoid recreation overhead.

        Args:
            eval_dataset: Evaluation dataset to use. Can be:
                - None: Uses self.eval_dataset
                - str: Uses self.eval_dataset[eval_dataset] (for named eval sets)
                - Dataset: Overrides self.eval_dataset directly

        Returns:
            Configured and accelerator-prepared evaluation DataLoader.

        Raises:
            ValueError: If both eval_dataset and self.eval_dataset are None.

        Note:
            When persistent_workers is True, DataLoaders are cached in
            self._eval_dataloaders to avoid worker process recreation between
            evaluation calls. The cache key is the dataset name (if string)
            or "eval" for the default dataset.
        """
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
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
