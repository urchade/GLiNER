from typing import Optional
from dataclasses import dataclass, field

import transformers
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    others_lr: Optional[float] = None
    others_weight_decay: Optional[float] = 0.0
    focal_loss_alpha: Optional[float] = -1
    focal_loss_gamma: Optional[float] = 0
    label_smoothing: Optional[float] = 0
    loss_reduction: Optional[str] = 'sum' 

class Trainer(transformers.Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        self.model.save_pretrained(output_dir)
        
    def compute_loss(self, model, inputs):
        """
        Override compute_loss to use a custom loss function.
        """
        # Forward pass
        outputs = model(alpha = self.args.focal_loss_alpha,
                        gamma = self.args.focal_loss_gamma,
                        label_smoothing = self.args.label_smoothing,
                        reduction = self.args.loss_reduction,
                        **inputs)
        loss = outputs.loss
        return loss
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.others_lr is not None:
                encoder_parameters = [name for name, _ in opt_model.named_parameters() if "encoder" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.others_weight_decay,
                        "lr": self.args.others_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.others_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
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

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer