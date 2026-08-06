from pathlib import Path
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
import logging
from PIL import Image
from torchvision import transforms

from ..data.dataset import _build_transform
from ..models.classifier import LayoutClassifier

logger = logging.getLogger(__name__)


class LayoutPredictor:
    """版式分类器推理模块。

    加载训练好的 checkpoint，对单张或批量图片进行版式类别预测。
    支持温度缩放校准与置信度拒绝机制：当最高置信度低于
    ``confidence_threshold`` 或 top1-top2 间隔低于 ``margin_threshold``
    时，结果归为"其他"（rejected=True，reason 说明原因）。
    阈值默认取自 checkpoint 内的 ``calibration`` 字段（训练时自动生成），
    也可在 predict 调用时临时覆盖。

    Usage::

        predictor = LayoutPredictor("output/best_model.pth", device="cpu")
        result = predictor.predict("test.jpg")
        print(result["category"], result["confidence"], result["rejected"])
    """

    REJECT_CATEGORY = "其他"

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
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

        # 路径兜底：训练机上的模型路径在部署环境不可用时，回退到
        # CLIP_MODELS_DIR 环境变量或容器内默认路径（/shared/models）
        if not model_path.exists():
            for candidate in (os.environ.get("CLIP_MODELS_DIR"), "/shared/models"):
                if candidate and Path(candidate).exists():
                    model_path = Path(candidate)
                    logger.info(f"Model path {raw_path!r} not found, falling back to {model_path}")
                    break

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

        calib = checkpoint.get("calibration") or {}
        # 参数优先级：构造参数 > checkpoint 内置 > 默认值
        self.temperature = temperature if temperature is not None else calib.get("temperature", 1.0)
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None
            else calib.get("confidence_threshold", 0.6)
        )
        self.margin_threshold = (
            margin_threshold if margin_threshold is not None
            else calib.get("margin_threshold", 0.1)
        )
        if self.temperature <= 0:
            self.temperature = 1.0

        self.transform = _build_transform(is_train=False)

    def predict(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        confidence_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """单张图片预测。

        Args:
            image: 图片路径、PIL Image 或已预处理的 Tensor [C, H, W]。
            confidence_threshold: 临时覆盖绝对置信度阈值。
            margin_threshold: 临时覆盖 top1-top2 间隔阈值。
            temperature: 临时覆盖温度（logits 除以 T 后 softmax）。

        Returns:
            dict with keys: category, index, confidence, probabilities,
            rejected, reason, original_category, original_index。
            被拒绝时 category 为 ``REJECT_CATEGORY``（其他），
            index 为 -1，original_* 保留原始 top-1 信息。
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        t = temperature if temperature is not None else self.temperature
        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits / t, dim=1)
            conf, idx = probs.max(dim=1)
            idx_i = idx.item()
            conf_i = conf.item()

        rejected, reason = self._decide(probs, conf_i, idx_i, confidence_threshold, margin_threshold)
        category = self.REJECT_CATEGORY if rejected else self.categories[idx_i]

        return {
            "category": category,
            "index": -1 if rejected else idx_i,
            "confidence": round(conf_i, 6),
            "probabilities": {
                cat: round(p, 6)
                for cat, p in zip(self.categories, probs[0].tolist())
            },
            "rejected": rejected,
            "reason": reason,
            "original_category": self.categories[idx_i],
            "original_index": idx_i,
        }

    def _decide(
        self,
        probs: torch.Tensor,
        top1_conf: float,
        top1_idx: int,
        confidence_threshold: Optional[float],
        margin_threshold: Optional[float],
    ) -> tuple[bool, Optional[str]]:
        """根据绝对置信度与 top1-top2 间隔判定是否拒绝。

        Returns:
            (rejected, reason)，reason 为 "low_confidence" / "ambiguous" / None。
        """
        ct = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        mt = self.margin_threshold if margin_threshold is None else margin_threshold

        if ct is not None and top1_conf < ct:
            return True, "low_confidence"
        if mt is not None and probs.size(1) >= 2:
            top2_conf = probs.topk(2, dim=1).values[0, 1].item()
            if top1_conf - top2_conf < mt:
                return True, "ambiguous"
        return False, None

    def predict_batch(self, images: list[Union[str, Image.Image]]) -> list[dict]:
        """批量图片预测。"""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def predict_top_k(
        self,
        image: Union[str, Image.Image],
        k: int = 2,
        temperature: Optional[float] = None,
    ) -> list[dict]:
        """返回 Top-K 分类结果（使用校准温度的概率）。"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        t = temperature if temperature is not None else self.temperature
        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits / t, dim=1)

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
