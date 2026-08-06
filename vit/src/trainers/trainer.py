import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Trainer:
    """版式分类器训练器。

    负责训练循环、验证、checkpoint 管理和早停。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: Optional[torch.device] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.training_cfg = config["training"]
        self.output_cfg = config["output"]
        self.calibration_cfg = config.get("calibration", {})
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.training_cfg["lr"],
            weight_decay=self.training_cfg.get("weight_decay", 0.0),
        )
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.scaler = torch.amp.GradScaler("cuda") if self.training_cfg.get("mixed_precision") else None

        lr_scheduler = self.training_cfg.get("lr_scheduler")
        if lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_cfg["epochs"],
            )
        elif lr_scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=max(1, self.training_cfg["epochs"] // 3),
                gamma=0.1,
            )
        else:
            self.scheduler = None

        self.output_dir = Path(self.output_cfg["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict] = []

    def train(self) -> None:
        epochs = self.training_cfg["epochs"]
        patience = self.training_cfg.get("early_stop_patience", epochs)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(epoch, is_train=True)
            val_loss, val_acc = self._run_epoch(epoch, is_train=False)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(record)

            self._maybe_save_checkpoint(val_acc, epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break

        self._save_history()
        print(f"Training complete. Best checkpoint saved at epoch {self.best_epoch}")

    def _run_epoch(self, epoch: int, is_train: bool) -> tuple[float, float]:
        if is_train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        total_loss = 0.0
        all_preds = []
        all_labels = []
        log_interval = self.training_cfg.get("log_interval", 10)
        mode = "Train" if is_train else "Val  "

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if is_train:
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if batch_idx % log_interval == 0:
                print(f"  [{mode}] Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} | Loss {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        print(f"  [{mode}] Epoch {epoch:3d} | Loss {avg_loss:.4f} | Acc {acc:.4f}")
        return avg_loss, acc

    def _maybe_save_checkpoint(self, val_acc: float, epoch: int) -> None:
        save_best_only = self.training_cfg.get("save_best_only", True)

        if save_best_only and val_acc <= self.best_val_acc:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "categories": list(self.model.idx_to_category.values()),
            "config": self.cfg,
        }

        path = self.output_dir / self.output_cfg.get("model_name", "checkpoint.pth")
        torch.save(checkpoint, path)
        if val_acc > self.best_val_acc:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def _save_history(self) -> None:
        path = self.output_dir / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def evaluate(self) -> dict:
        """在验证集上做完整评估，返回分类报告和混淆矩阵。"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        category_names = [self.model.idx_to_category.get(i, f"class_{i}") for i in range(self.model.num_classes)]

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "report": classification_report(all_labels, all_preds, target_names=category_names, zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            "labels": category_names,
        }

    # ------------------------------------------------------------------
    # 温度校准与置信度拒绝阈值
    # ------------------------------------------------------------------
    def calibrate(self) -> dict:
        """在验证集上执行温度校准并生成置信度拒绝阈值。

        流程：
        1. 加载 best_model.pth 权重，在验证集上收集 logits。
        2. 拟合温度 T（最小化校准后概率的负对数似然）。
        3. 用校准后概率统计正确样本的 top-1 置信度与 top1-top2 间隔，
           取其 ``percentile`` 分位数乘 ``safety_factor`` 作为阈值
           （保证绝大多数正确样本不会被误拒）。
        4. 校准参数写回 checkpoint（best_model.pth 与当前模型文件），
           并生成 calibration_report.json。

        Returns:
            校准参数字典（无 best checkpoint 时返回空 dict）。
        """
        best_path = self.output_dir / "best_model.pth"
        if not best_path.exists():
            print("  [Calibrate] No best_model.pth found, skip calibration")
            return {}

        ccfg = self.calibration_cfg
        percentile = float(ccfg.get("percentile", 5))
        safety_factor = float(ccfg.get("safety_factor", 0.9))
        t_bounds = ccfg.get("temperature_bounds", [0.1, 10.0])
        conf_min = float(ccfg.get("confidence_threshold_min", 0.3))
        conf_max = float(ccfg.get("confidence_threshold_max", 0.9))
        margin_min = float(ccfg.get("margin_threshold_min", 0.02))
        margin_max = float(ccfg.get("margin_threshold_max", 0.5))

        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                logits_list.append(self.model(images).cpu())
                labels_list.append(labels)
        logits = torch.cat(logits_list).float()
        labels = torch.cat(labels_list)

        def nll(t: float) -> float:
            scaled = logits / max(float(t), 1e-6)
            log_probs = F.log_softmax(scaled, dim=1)
            return -log_probs.gather(1, labels.unsqueeze(1)).mean().item()

        nll_before = nll(1.0)
        temperature = self._fit_temperature(nll, t_bounds)
        nll_after = nll(temperature)

        probs = F.softmax(logits / temperature, dim=1)
        top1, preds = probs.max(dim=1)
        if probs.size(1) >= 2:
            top2 = probs.topk(2, dim=1).values[:, 1]
        else:
            top2 = torch.full_like(top1, float("inf"))
        margins = top1 - top2

        correct_mask = preds == labels
        n_correct = int(correct_mask.sum().item())

        if n_correct > 0:
            conf_q = torch.quantile(top1[correct_mask], percentile / 100.0).item()
            margin_q = torch.quantile(margins[correct_mask], percentile / 100.0).item()
        else:
            conf_q = margin_q = 0.0

        confidence_threshold = min(max(conf_q * safety_factor, conf_min), conf_max)
        margin_threshold = min(max(margin_q * safety_factor, margin_min), margin_max)

        calibration = {
            "temperature": round(float(temperature), 6),
            "confidence_threshold": round(float(confidence_threshold), 6),
            "margin_threshold": round(float(margin_threshold), 6),
            "percentile": percentile,
            "safety_factor": safety_factor,
            "n_val_samples": int(len(labels)),
            "n_correct": n_correct,
            "accuracy": round(float(n_correct) / max(len(labels), 1), 6),
            "nll_before": round(nll_before, 6),
            "nll_after": round(nll_after, 6),
            "p5_conf_raw": round(float(conf_q), 6),
            "p5_margin_raw": round(float(margin_q), 6),
        }

        for path in (best_path, self.output_dir / self.output_cfg.get("model_name", "checkpoint.pth")):
            try:
                c = torch.load(path, map_location="cpu", weights_only=False)
                c["calibration"] = calibration
                torch.save(c, path)
            except Exception as e:
                print(f"  [Calibrate] Failed to update {path}: {e}")

        self._save_calibration_report(calibration, probs, preds, labels, margins)

        print("\n=== Calibration ===")
        print(f"  Temperature:       {temperature:.4f} (NLL {nll_before:.4f} -> {nll_after:.4f})")
        print(f"  Conf threshold:    {confidence_threshold:.4f} (P{percentile:g} raw {conf_q:.4f})")
        print(f"  Margin threshold:  {margin_threshold:.4f} (P{percentile:g} raw {margin_q:.4f})")
        print(f"  Val accuracy:      {calibration['accuracy']:.4f} ({n_correct}/{len(labels)})")

        return calibration

    @staticmethod
    def _fit_temperature(nll_fn, bounds: list[float]) -> float:
        """在 [lo, hi] 内最小化 NLL 拟合温度。优先 scipy，失败则网格搜索。"""
        lo, hi = float(bounds[0]), float(bounds[1])
        lo, hi = max(lo, 0.05), min(hi, 50.0)
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(nll_fn, bounds=(lo, hi), method="bounded")
            if res.success and lo <= res.x <= hi:
                return float(res.x)
        except Exception:
            pass
        best_t, best_v = lo, nll_fn(lo)
        for t in torch.linspace(lo, hi, 51).tolist():
            v = nll_fn(t)
            if v < best_v:
                best_t, best_v = t, v
        return float(best_t)

    def _save_calibration_report(self, calibration: dict, probs, preds, labels, margins) -> None:
        """生成分位数表与覆盖率/精度曲线报告。"""
        correct = preds == labels
        confs = probs.max(dim=1).values
        top1_conf = confs[correct]
        margins_ok = margins[correct]

        percentiles = [1, 5, 10, 25, 50, 75, 90]
        report = {"calibration": calibration}
        if len(top1_conf) > 0:
            report["confidence_percentiles"] = {
                f"p{p}": round(float(torch.quantile(top1_conf, p / 100.0).item()), 6)
                for p in percentiles
            }
            report["margin_percentiles"] = {
                f"p{p}": round(float(torch.quantile(margins_ok, p / 100.0).item()), 6)
                for p in percentiles
            }

        curve = []
        for t in torch.linspace(0.0, 1.0, 21).tolist():
            kept = confs >= t
            n_kept = int(kept.sum().item())
            if n_kept > 0:
                acc = float((preds[kept] == labels[kept]).float().mean().item())
            else:
                acc = 0.0
            curve.append({
                "threshold": round(float(t), 2),
                "coverage": round(n_kept / max(len(confs), 1), 4),
                "accuracy": round(acc, 4),
            })
        report["coverage_curve"] = curve

        path = self.output_dir / "calibration_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  [Calibrate] Report saved: {path}")
