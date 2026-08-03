from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..data.dataset import _build_transform
from ..models.classifier import LayoutClassifier


class LayoutPredictor:
    """版式分类器推理模块。

    加载训练好的 checkpoint，对单张或批量图片进行版式类别预测。

    Usage::

        predictor = LayoutPredictor("output/best_model.pth", device="cpu")
        result = predictor.predict("test.jpg")
        print(result["category"], result["confidence"])
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = checkpoint["config"]
        categories = checkpoint["categories"]
        model_cfg = self.cfg["model"]
        raw_path = model_cfg["path"]
        model_path = Path(raw_path)
        if not model_path.is_absolute():
            model_path = (checkpoint_path.parent / model_path).resolve()

        self.model = LayoutClassifier(
            model_path=str(model_path),
            num_classes=len(categories),
            freeze_encoder=model_cfg.get("freeze_encoder", True),
            dropout=model_cfg.get("dropout", 0.1),
            projection_dim=model_cfg.get("projection_dim"),
            pool=model_cfg.get("pool", "cls"),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.set_categories(categories)
        self.categories = categories

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.transform = _build_transform(is_train=False)

    def predict(self, image: Union[str, Image.Image, torch.Tensor]) -> dict:
        """单张图片预测。

        Args:
            image: 图片路径、PIL Image 或已预处理的 Tensor [C, H, W]。

        Returns:
            dict with keys: category, confidence, index, probabilities
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            idx = idx.item()
            conf = conf.item()

        return {
            "category": self.categories[idx],
            "index": idx,
            "confidence": round(conf, 6),
            "probabilities": {
                cat: round(p, 6)
                for cat, p in zip(self.categories, probs[0].tolist())
            },
        }

    def predict_batch(self, images: list[Union[str, Image.Image]]) -> list[dict]:
        """批量图片预测。"""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def predict_top_k(self, image: Union[str, Image.Image], k: int = 2) -> list[dict]:
        """返回 Top-K 分类结果。"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits, dim=1)

        top_probs, top_indices = probs.topk(min(k, len(self.categories)), dim=1)

        results = []
        for i in range(top_indices.size(1)):
            idx = top_indices[0, i].item()
            results.append({
                "category": self.categories[idx],
                "index": idx,
                "confidence": round(top_probs[0, i].item(), 6),
            })
        return results
