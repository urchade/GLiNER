import argparse
import json
import os
import re
from types import SimpleNamespace

from tqdm import tqdm
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_constant_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup,
                          get_inverse_sqrt_schedule)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from gliner import GLiNER
from gliner.modules.base import load_config_as_namespace
from gliner.modules.run_evaluation import get_for_all_path


def save_top_k_checkpoints(model: GLiNER, save_path: str, checkpoint: int, top_k: int = 5):
    """
    Save the top-k checkpoints (latest k checkpoints) of a model and tokenizer.

    Parameters:
        model (GLiNER): The model to save.
        save_path (str): The directory path to save the checkpoints.
        top_k (int): The number of top checkpoints to keep. Defaults to 5.
    """
    # Save the current model and tokenizer
    if isinstance(model, DDP):
        model.module.save_pretrained(os.path.join(save_path, checkpoint))
    else:
        model.save_pretrained(os.path.join(save_path, checkpoint))
    # tokenizer.save_pretrained(save_path)

    # List all files in the directory
    files = os.listdir(save_path)

    # Filter files to keep only the model checkpoints
    checkpoint_folders = [file for file in files if re.search('model\\_\\d+', file)]

    # Sort checkpoint files by modification time (latest first)
    checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)

    # Keep only the top-k checkpoints
    for checkpoint_folder in checkpoint_folders[top_k:]:
        checkpoint_folder = os.path.join(save_path, checkpoint_folder)
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder)]
        for file in checkpoint_files:
            os.remove(file)
        os.rmdir(os.path.join(checkpoint_folder))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def setup_model_and_optimizer(model_config, lr_encoder, lr_others, 
                              weight_decay_encoder, weight_decay_others, 
                              freeze_token_rep, rank, device):
    model = GLiNER(model_config).to(device)
    
    if rank is not None:
        ddp_model = DDP(model, device_ids=[rank],output_device=rank, find_unused_parameters=False)
    else:
        ddp_model = model

    optimizer = ddp_model.module.get_optimizer(lr_encoder, lr_others,
                                        weight_decay_encoder, weight_decay_others,
                                        freeze_token_rep=freeze_token_rep)
    return ddp_model, optimizer


def train_dist(rank, world_size, dataset, train_batch_size, model_config, lr_encoder, lr_others,
                weight_decay_encoder, weight_decay_others, freeze_token_rep, num_steps, *args, **kwargs):
    setup(rank, world_size)
    model, optimizer = setup_model_and_optimizer(model_config, lr_encoder, lr_others, 
                                                 weight_decay_encoder, weight_decay_others, 
                                                 freeze_token_rep, rank, device=f'cuda:{rank}')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_loader = model.module.create_dataloader(dataset, batch_size=train_batch_size, shuffle=False, sampler=sampler)

    num_steps //= world_size
    train(model, optimizer, train_loader, train_batch_size, num_steps, rank = rank, *args, **kwargs)
    cleanup()


def train(model, optimizer, train_loader, train_batch_size, num_steps=1000, eval_every=100, log_dir="logs", val_data_dir="none",
          warmup_ratio=0.1, scheduler_type="cosine", save_total_limit=5, sampler=None, device='cuda', rank=None):
    model.train()
    pbar = tqdm(range(num_steps))

    num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    elif scheduler_type == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif scheduler_type == "inverse_sqrt":
        scheduler = get_inverse_sqrt_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        raise ValueError(
            f"Invalid sheduler_type value: '{scheduler_type}' \n Supported scheduler types: 'cosine', 'linear', 'constant', 'polynomial', 'inverse_sqrt'"
        )

    iter_train_loader = iter(train_loader)
    scaler = torch.cuda.amp.GradScaler()

    for step in pbar:
        optimizer.zero_grad()

        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(x)

            if torch.isnan(loss).any():
                print("Warning: NaN loss detected")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        except Exception as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            continue

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % eval_every == 0:
            if rank is None or rank == 0:
                checkpoint = f'model_{step + 1}'
                save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
                if val_data_dir != "none":
                    get_for_all_path(model, step, log_dir, val_data_dir)
                model.train()


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    parser.add_argument('--allow_distributed', type=bool, default= True, help='Whether to allow distributed training if there are more than one GPU available')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = load_config_as_namespace(args.config)
    config.log_dir = args.log_dir

    with open(config.train_data, 'r') as f:
        data = json.load(f)

    model_config = SimpleNamespace(
        model_name=config.model_name,
        name=config.name,
        max_width=config.max_width,
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        fine_tune=config.fine_tune,
        subtoken_pooling=config.subtoken_pooling,
        span_mode=config.span_mode,
        loss_alpha=config.loss_alpha,
        loss_gamma=config.loss_gamma,
        loss_reduction=config.loss_reduction,
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)
    weight_decay_encoder = float(config.weight_decay_encoder)
    weight_decay_others = float(config.weight_decay_other)

    if torch.cuda.device_count() > 1 and args.allow_distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train_dist, args=(world_size, data, config.train_batch_size, model_config, lr_encoder, lr_others, weight_decay_encoder, weight_decay_others, 
                                   config.freeze_token_rep, config.num_steps, config.eval_every, config.log_dir, 
                                   config.val_data_dir, config.warmup_ratio, config.scheduler_type, 
                                   config.save_total_limit), nprocs=world_size, join=True)
    else:
        if config.prev_path != "none":
            model = GLiNER.from_pretrained(config.prev_path)
            model.config = config
        else:
            model = GLiNER(model_config)

        model = model.to(device)

        optimizer = model.get_optimizer(lr_encoder, lr_others,
                                        weight_decay_encoder, weight_decay_others,
                                        freeze_token_rep=config.freeze_token_rep)
            
        train_loader = model.create_dataloader(data, batch_size=config.train_batch_size, shuffle=True)

        train(model, optimizer, train_loader, num_steps=config.num_steps, eval_every=config.eval_every,
              log_dir=config.log_dir, val_data_dir=config.val_data_dir, warmup_ratio=config.warmup_ratio,
              train_batch_size=config.train_batch_size, scheduler_type=config.scheduler_type, 
              save_total_limit=config.save_total_limit, device=device)
