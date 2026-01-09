import argparse
import json
from pathlib import Path
from gliner import GLiNER
from gliner.utils import load_config_as_namespace, namespace_to_dict


def load_json_data(path: str):
    """Load JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_cfg: dict, train_cfg: dict):
    """Build or load GLiNER model."""
    prev_path = train_cfg.get("prev_path")
    if prev_path and str(prev_path).lower() not in ("none", "null", ""):
        print(f"Loading pretrained model from: {prev_path}")
        return GLiNER.from_pretrained(prev_path)
    print("Initializing model from config...")
    return GLiNER.from_config(model_cfg)


def main(cfg_path: str):
    """Main training function."""
    # Load config
    cfg = load_config_as_namespace(cfg_path)

    # Convert to dicts for model building
    model_cfg = namespace_to_dict(cfg.model)
    train_cfg = namespace_to_dict(cfg.training)

    # Setup output directory
    output_dir = Path(cfg.data.root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading training data from: {cfg.data.train_data}")
    train_dataset = load_json_data(cfg.data.train_data)
    print(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if hasattr(cfg.data, "val_data_dir") and cfg.data.val_data_dir.lower() not in ("none", "null", ""):
        print(f"Loading validation data from: {cfg.data.val_data_dir}")
        eval_dataset = load_json_data(cfg.data.val_data_dir)
        print(f"Validation samples: {len(eval_dataset)}")

    # Build model
    model = build_model(model_cfg, train_cfg)
    print(f"Model type: {model.__class__.__name__}")
    
    # Get freeze components
    freeze_components = train_cfg.get("freeze_components", None)
    if freeze_components:
        print(f"Freezing components: {freeze_components}")

    # Train
    print("\nStarting training...")
    model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="models",
        # Schedule
        max_steps=cfg.training.num_steps,
        lr_scheduler_type=cfg.training.scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        # Batch & optimization
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.train_batch_size,
        learning_rate=float(cfg.training.lr_encoder),
        others_lr=float(cfg.training.lr_others),
        weight_decay=float(cfg.training.weight_decay_encoder),
        others_weight_decay=float(cfg.training.weight_decay_other),
        max_grad_norm=float(cfg.training.max_grad_norm),
        # Loss
        focal_loss_alpha=float(cfg.training.loss_alpha),
        focal_loss_gamma=float(cfg.training.loss_gamma),
        focal_loss_prob_margin=float(getattr(cfg.training, "loss_prob_margin", 0.0)),
        loss_reduction=cfg.training.loss_reduction,
        negatives=float(cfg.training.negatives),
        masking=cfg.training.masking,
        # Logging & saving
        save_steps=cfg.training.eval_every,
        logging_steps=cfg.training.eval_every,
        save_total_limit=cfg.training.save_total_limit,
        # Freezing
        freeze_components=freeze_components,
    )

    print(f"\n✓ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLiNER model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file (YAML or JSON)")
    args = parser.parse_args()
    main(args.config)
