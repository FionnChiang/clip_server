#!/usr/bin/env python3
"""环境验证脚本：检查 GPU、加载模型、跑一次前向传播。

不依赖训练 checkpoint，只验证基础环境是否正常。

Usage:
    docker run --rm --gpus all \
      -v $(pwd):/workspace \
      layout-classifier:latest \
      python scripts/verify.py
"""

import os
import sys
from pathlib import Path
import argparse

current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

import torch


def check_gpu() -> dict:
    result = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["cuda_version"] = torch.version.cuda
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1024**3, 1
        )
        dummy = torch.randn(2, 3, 224, 224).cuda()
        _ = dummy * 2
    return result


def check_model(model_path: str) -> dict:
    from src.models.classifier import LayoutClassifier

    result = {}
    result["model_path"] = model_path
    result["model_path_exists"] = Path(model_path).is_dir()

    model = LayoutClassifier(
        model_path=model_path,
        num_classes=2,
        freeze_encoder=True,
    )
    result["total_params"] = model.total_params()
    result["trainable_params"] = model.trainable_params()
    result["encoder_frozen"] = model.is_encoder_frozen()

    dummy = torch.randn(2, 3, 224, 224)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy.to(device))
    result["forward_output_shape"] = list(out.shape)
    result["forward_passed"] = True
    return result


def check_dataset(data_root: str, categories: list[str]) -> dict:
    from src.data.dataset import build_dataloaders

    result = {}
    if not Path(data_root).is_dir():
        result["data_root_exists"] = False
        result["error"] = f"Data root not found: {data_root}"
        return result

    try:
        _, _, num_classes, weights = build_dataloaders(
            data_root=data_root,
            categories=categories,
            batch_size=4,
            train_ratio=0.8,
            seed=42,
            num_workers=0,
            class_balance="weighted_loss",
        )
        result["num_classes"] = num_classes
        result["data_root_exists"] = True
        if weights is not None:
            result["class_weights"] = dict(zip(categories, [round(w.item(), 3) for w in weights]))
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify training environment")
    parser.add_argument("--model-path", type=str, default="models")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--categories", type=str, nargs="*", default=[])
    args = parser.parse_args()

    print("=" * 50)
    print("  GPU Check")
    print("=" * 50)
    gpu = check_gpu()
    for k, v in gpu.items():
        print(f"  {k:20s}: {v}")
    if not gpu["cuda_available"]:
        print("\n  [WARN] CUDA not available, will run on CPU.")

    print("\n" + "=" * 50)
    print("  Model Check")
    print("=" * 50)
    model = check_model(args.model_path)
    for k, v in model.items():
        print(f"  {k:20s}: {v}")

    if args.data_root and args.categories:
        print("\n" + "=" * 50)
        print("  Dataset Check")
        print("=" * 50)
        ds = check_dataset(args.data_root, args.categories)
        for k, v in ds.items():
            print(f"  {k:20s}: {v}")

    all_passed = gpu["cuda_available"] and model.get("forward_passed")
    print("\n" + "=" * 50)
    print(f"  Overall: {'PASSED' if all_passed else 'WARNINGS'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
