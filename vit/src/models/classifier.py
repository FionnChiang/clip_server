from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPConfig


def _load_vision_encoder(model_path: str) -> CLIPVisionModel:
    """从完整 CLIP checkpoint 中加载仅视觉编码器。

    完整 CLIP checkpoint 的 state_dict 中视觉权重键名以 ``vision_model.`` 为前缀，
    需要剥离前缀后才能正确加载到 CLIPVisionModel 中。
    """
    model_path = Path(model_path)

    config = CLIPConfig.from_pretrained(str(model_path))
    vision_config = config.vision_config

    model = CLIPVisionModel(vision_config)

    weight_file = model_path / "pytorch_model.bin"
    if not weight_file.exists():
        weight_file = model_path / "model.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"No model weights found in {model_path}")

    if weight_file.suffix == ".bin":
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))

    vision_state_dict = {}
    prefix = "vision_model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            vision_state_dict[key[len(prefix):]] = value

    if not vision_state_dict:
        available = [k for k in state_dict if "vision" in k.lower()][:5]
        raise RuntimeError(
            f"No vision_model weights found in checkpoint. "
            f"First vision-related keys: {available}"
        )

    missing, unexpected = model.load_state_dict(vision_state_dict, strict=False)
    if missing:
        print(f"  [Warn] Missing vision encoder keys: {missing[:3]}...")
    if unexpected:
        print(f"  [Warn] Unexpected vision encoder keys: {unexpected[:3]}...")

    return model


class LayoutClassifier(nn.Module):
    """基于 CLIP ViT 的版式分类器。

    在冻结的 CLIP 视觉编码器之上添加分类头，用于版式类别预测。

    Args:
        model_path: CLIP 模型文件目录路径。
        num_classes: 分类类别数。
        freeze_encoder: 是否冻结 ViT 编码器参数。
        dropout: 分类头 dropout 概率。
        projection_dim: 分类头中间层维度，为 None 则直接 Linear(num_classes)。
        pool: 特征池化方式，支持 "cls" 和 "mean"。
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        projection_dim: Optional[int] = None,
        pool: str = "cls",
    ):
        super().__init__()

        model_path = str(Path(model_path).resolve())
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self.vision_encoder = _load_vision_encoder(model_path)

        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        hidden_dim = self.vision_encoder.config.hidden_size
        self.pool = pool

        if projection_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        self.num_classes = num_classes
        self.idx_to_category: dict[int, str] = {}

    def set_categories(self, categories: list[str]) -> None:
        self.idx_to_category = {i: cat for i, cat in enumerate(categories)}

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.is_encoder_frozen()):
            outputs = self.vision_encoder(pixel_values=pixel_values)

        if self.pool == "cls":
            features = outputs.last_hidden_state[:, 0, :]
        elif self.pool == "mean":
            features = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool mode: {self.pool}")

        return self.classifier(features)

    def is_encoder_frozen(self) -> bool:
        """检查编码器是否冻结。"""
        return not any(p.requires_grad for p in self.vision_encoder.parameters())

    def trainable_params(self) -> int:
        """可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        """总参数量。"""
        return sum(p.numel() for p in self.parameters())
