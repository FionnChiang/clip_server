import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
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
