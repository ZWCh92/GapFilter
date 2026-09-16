"""
定义网络结构, 实现从 feature 到表达的映射。

loss 使用 MAE(真实梯度, 预测梯度) + 保证方向性。

输入：
1. 从训练集上获取位置相近的两个 spot 的 H&E feature，分别记为 x1, x2；
2. 对应 spot 的待预测基因的 min-max 后的表达量，记为 y1, y2；

输出：
1. 预测的后面 spot 表达量 - 前面 spot 表达量，记为 y_pred = f(x1, x2)；
2. 拟合 MAE(y2 - y1, y_pred) + k * MAE(f(x1, x2) + f(x2, x1), 0)。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class GradientMLP(nn.Module):
    """f(x1, x2) -> Δy，输入为 [x1; x2]。"""

    def __init__(
        self,
        feature_dim: int,
        gene_dim: int,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = feature_dim * 2
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, gene_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x1, x2], dim=-1)
        return self.net(z)


class GapFilterModel(pl.LightningModule):
    """配对 spot 表达梯度预测模型。

    期望 batch 字段与 ``GapFilterPairDataset`` 一致：
    ``x1, x2, y1, y2``。
    """

    def __init__(self, config: Any = None):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        feature_dim = int(_cfg_get(config, "feature_dim", 3072))
        gene_dim = int(_cfg_get(config, "gene_dim", 200))
        hidden_dims = tuple(_cfg_get(config, "hidden_dims", (1024, 512)))
        dropout = float(_cfg_get(config, "dropout", 0.1))
        self.antisym_weight = float(_cfg_get(config, "antisym_weight", 1.0))
        self.lr = float(_cfg_get(config, "lr", 1e-3))
        self.weight_decay = float(_cfg_get(config, "weight_decay", 1e-4))

        self.net = GradientMLP(
            feature_dim=feature_dim,
            gene_dim=gene_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """预测 Δy = y2 - y1。"""
        return self.net(x1, x2)

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> dict:
        y_pred = self(x1, x2)
        y_true = y2 - y1
        loss_grad = F.l1_loss(y_pred, y_true)

        y_pred_rev = self(x2, x1)
        loss_antisym = F.l1_loss(y_pred + y_pred_rev, torch.zeros_like(y_pred))

        loss = loss_grad + self.antisym_weight * loss_antisym
        return {
            "loss": loss,
            "loss_grad": loss_grad.detach(),
            "loss_antisym": loss_antisym.detach(),
            "y_pred": y_pred,
        }

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        out = self.compute_loss(batch["x1"], batch["x2"], batch["y1"], batch["y2"])
        self.log(
            f"{stage}/loss",
            out["loss"],
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x1"].size(0),
        )
        self.log(
            f"{stage}/loss_grad",
            out["loss_grad"],
            on_step=False,
            on_epoch=True,
            batch_size=batch["x1"].size(0),
        )
        self.log(
            f"{stage}/loss_antisym",
            out["loss_antisym"],
            on_step=False,
            on_epoch=True,
            batch_size=batch["x1"].size(0),
        )
        return out["loss"]

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "test")

    def predict_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        y_pred = self(batch["x1"], batch["x2"])
        return {
            "y_pred": y_pred,
            "y_true": batch["y2"] - batch["y1"],
            "idx1": batch.get("idx1"),
            "idx2": batch.get("idx2"),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(_cfg_get(self.config, "max_epochs", 100)),
            eta_min=float(_cfg_get(self.config, "lr_min", 1e-6)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
