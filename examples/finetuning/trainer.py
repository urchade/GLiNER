from contextlib import nullcontext, contextmanager
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from beartype import beartype
from beartype.typing import Optional
from gliner import GLiNER
from torch.nn import Module
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def divisible_by(num, den):
    return (num % den) == 0


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters=True
)


class GlinerTrainer(Module):
    @beartype
    def __init__(
            self,
            model,
            train_data: list,
            batch_size: int,

            grad_accum_every: Optional[int] = None,
            max_types: int = 25,  # maximum number of entity types during training
            shuffle_types: bool = True,  # if shuffle or not entity types
            random_drop: bool = True,  # randomly drop entity types
            max_neg_type_ratio: int = 1,
            # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
            max_len: int = 384,  # maximum sentence length
            lr_encoder: float = 1e-5,
            lr_others: float = 5e-5,

            freeze_token_rep: bool = False,
            val_data: Optional[dict] = None,
            val_every_step: Optional[int] = None,
            accelerator_kwargs: dict = dict(),
            optimizer_kwargs: dict = dict(),
            checkpoint_every_step: Optional[int] = None,
            checkpoint_every_epoch: Optional[int] = None,
            checkpoint_folder='./checkpoints',
            use_wandb_tracking=False
    ):
        super().__init__()
        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        self.model = model

        self.model.set_sampling_params(
            max_types=max_types,
            shuffle_types=shuffle_types,
            random_drop=random_drop,
            max_neg_type_ratio=max_neg_type_ratio,
            max_len=max_len
        )
        self.optimizer = model.get_optimizer(lr_encoder, lr_others, freeze_token_rep=freeze_token_rep,
                                             weight_decay_encoder=1e-2, weight_decay_others=1e-2,
                                             **optimizer_kwargs)
        self.train_loader = model.create_dataloader(train_data, batch_size=batch_size, shuffle=True)

        self.should_validate = exists(val_data)
        if self.should_validate:
            assert len(val_data) > 0, 'your validation dataset is empty'
            self.val_every_step = val_every_step
            self.val_data = val_data

        (
            self.optimizer,
            self.model,
            self.train_loader
        ) = self.accelerator.prepare(
            self.optimizer,
            self.model,
            self.train_loader
        )
        self.grad_accum_every = grad_accum_every if grad_accum_every else 64 // batch_size
        if not grad_accum_every:
            print(
                f"grad_accum_every not set, using {self.grad_accum_every} to achive a effective batch size of 64 (64 / {batch_size}) = {self.grad_accum_every}")

        self.register_buffer('step', torch.tensor(0))

        self.checkpoint_every_step = checkpoint_every_step
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

    @contextmanager
    @beartype
    def trackers(
            self,
            project_name: str,
            run_name: Optional[str] = None,
            hps: Optional[dict] = None
    ):
        assert self.use_wandb_tracking

        self.accelerator.init_trackers(project_name, config=hps)

        if exists(run_name):
            self.accelerator.trackers[0].run.name = run_name

        yield
        self.accelerator.end_training()

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step=self.step.item())

    @property
    def device(self):
        return self.unwrapped_model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()
        pkg = dict(
            model=self.unwrapped_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step.item()
        )
        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path))
        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    def train(self,
              num_epochs=None,
              num_steps=None,
              num_warmup_steps: int = 1000):
        assert not (num_epochs is None and num_steps is None), 'you must specify either num_epochs or num_steps'
        step = self.step.item()
        total_steps = num_epochs * len(self.train_loader) if num_steps is None else num_steps
        num_epochs = default(num_epochs, total_steps // len(self.train_loader))

        total_steps += default(step, 0)
        avg_epoch_loss = 0

        self.print(f"Training for {total_steps} steps which is {num_epochs} epochs.")
        epoch_size = len(self.train_loader)
        self.model.train()

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                                total=len(self.train_loader))

            for batch_idx, batch in progress_bar:
                is_last = (batch_idx + 1) % self.grad_accum_every == 0
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                with self.accelerator.autocast(), maybe_no_sync():
                    total_loss = self.model(batch)
                    self.accelerator.backward(total_loss / self.grad_accum_every)

                current_loss = total_loss.item()
                total_epoch_loss += current_loss

                step += 1
                self.step.add_(1)
                progress_bar.set_postfix(loss=current_loss, average_loss=max(1, total_epoch_loss) / (batch_idx + 1),
                                         step=step)

                self.scheduler.step()
                if is_last or (batch_idx + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if step > total_steps:
                    self.print(f'Finished training after {step} steps')
                    break

                if self.is_main and self.checkpoint_every_step is not None and step % self.checkpoint_every_step == 0:
                    self.model.save_pretrained(self.checkpoint_folder / f'GliNER.step_{step}')

                if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                    self.unwrapped_model.eval()
                    results, f1 = self.model.evaluate(self.val_data["samples"], flat_ner=True, threshold=0.5,
                                                      batch_size=12,
                                                      entity_types=self.val_data["entity_types"])

                    self.print(f"Step={step}\n{results}")
                    self.log(val_loss=f1)
                    self.unwrapped_model.train()

            avg_epoch_loss = total_epoch_loss / epoch_size
            epochOut = f'Epoch {epoch + 1} average loss: {avg_epoch_loss}'

            self.wait()
            self.print(epochOut)
            if self.is_main and self.checkpoint_every_epoch is not None and (
                    self.checkpoint_every_epoch == 1 or (epoch != 0 and epoch + 1 % self.checkpoint_every_epoch == 0)):
                self.model.save_pretrained(
                    self.checkpoint_folder / f'GliNER.epoch_{epoch}_avg_loss_{avg_epoch_loss:.3f}')

        return avg_epoch_loss
