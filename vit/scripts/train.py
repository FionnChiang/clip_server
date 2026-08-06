#!/usr/bin/env python3
"""训练入口脚本。

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path
import yaml

current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

import torch
from src.data.dataset import build_dataloaders
from src.models.classifier import LayoutClassifier
from src.trainers.trainer import Trainer


def _resolve_path(path: str, base_dir: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def main():
    parser = argparse.ArgumentParser(description="Layout Classifier Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = current_dir / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent
    data_root = _resolve_path(config["data"]["root"], base_dir)
    model_path = _resolve_path(config["model"]["path"], base_dir)
    output_dir = _resolve_path(config["output"]["dir"], base_dir)
    config["output"]["dir"] = output_dir

    categories = config["data"]["categories"]

    device = torch.device(args.device) if args.device else None

    print(f"Data root:      {data_root}")
    print(f"Model path:     {model_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Categories ({len(categories)}):     {categories}")
    print(f"Device:         {device or 'auto'}")

    train_loader, val_loader, num_classes, class_weights = build_dataloaders(
        data_root=data_root,
        categories=categories,
        batch_size=config["training"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        seed=config["data"]["seed"],
        num_workers=config["training"].get("num_workers", 4),
        class_balance=config["training"].get("class_balance", "none"),
    )

    print(f"Train samples:  {len(train_loader.dataset)}")
    print(f"Val samples:    {len(val_loader.dataset)}")

    model = LayoutClassifier(
        model_path=model_path,
        num_classes=num_classes,
        freeze_encoder=config["model"].get("freeze_encoder", True),
        dropout=config["model"].get("dropout", 0.1),
        projection_dim=config["model"].get("projection_dim"),
        pool=config["model"].get("pool", "cls"),
    )
    model.set_categories(categories)

    print(f"Total params:   {model.total_params():,}")
    print(f"Trainable:      {model.trainable_params():,}")
    print(f"Encoder frozen: {model.is_encoder_frozen()}")

    class_balance_mode = config["training"].get("class_balance", "none")
    print(f"Class balance:  {class_balance_mode}")
    if class_weights is not None:
        print(f"Class weights:  {dict(zip(categories, class_weights.tolist()))}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=class_weights,
    )
    trainer.train()

    print("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(eval_results["report"])

    if config.get("calibration", {}).get("enabled", True):
        calibration = trainer.calibrate()
        if calibration:
            print("\nCalibration parameters are embedded in the checkpoint.")


if __name__ == "__main__":
    main()
