import os
import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _build_transform(is_train: bool = False) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


class LayoutDataset(Dataset):
    """从文件夹结构中加载版式图片数据集。

    期望目录结构::

        data_root/
        ├── 身份证/
        │   ├── img_001.jpg
        │   └── img_002.jpg
        ├── 发票/
        │   ├── img_001.jpg
        │   └── ...
        └── ...

    每张图片根据其父文件夹名自动标注。
    """

    def __init__(
        self,
        data_root: str,
        categories: list[str],
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.categories = categories
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.idx_to_category = {i: cat for cat, i in self.category_to_idx.items()}

        self.samples: list[tuple[Path, int]] = []
        for category in categories:
            category_dir = self.data_root / category
            if not category_dir.is_dir():
                continue
            for img_path in category_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
                    self.samples.append((img_path, self.category_to_idx[category]))

        self.samples.sort(key=lambda x: x[0].name)
        random.seed(seed)
        random.shuffle(self.samples)

        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        elif split == "val":
            self.samples = self.samples[split_idx:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.transform = transform or _build_transform(is_train=(split == "train"))
        self.class_weights: Optional[torch.Tensor] = None

    def compute_class_weights(self) -> torch.Tensor:
        """计算类别权重（反频率归一化），用于加权损失。"""
        class_counts = [0] * len(self.categories)
        for _, label in self.samples:
            class_counts[label] += 1
        max_count = max(class_counts)
        weights = [max_count / max(c, 1) for c in class_counts]
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
        return self.class_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_dataloaders(
    data_root: str,
    categories: list[str],
    batch_size: int = 32,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 4,
    class_balance: str = "none",
) -> tuple[DataLoader, DataLoader, int, Optional[torch.Tensor]]:
    """构建训练集和验证集的 DataLoader。

    Args:
        class_balance: 类别均衡策略。
            - ``"none"``: 不做处理。
            - ``"weighted_loss"``: 计算反频率权重，交由 Trainer 注入 CrossEntropyLoss。
            - ``"oversample"``: 使用 WeightedRandomSampler 等概率采样每个类别。

    Returns:
        train_loader, val_loader, num_classes, class_weights
    """
    train_ds = LayoutDataset(
        data_root=data_root,
        categories=categories,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )
    val_ds = LayoutDataset(
        data_root=data_root,
        categories=categories,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
    )

    class_weights: Optional[torch.Tensor] = None

    if class_balance == "weighted_loss":
        class_weights = train_ds.compute_class_weights()
        sampler = None
        shuffle = True
    elif class_balance == "oversample":
        train_ds.compute_class_weights()
        sample_weights = [train_ds.class_weights[label].item() for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(categories), class_weights
