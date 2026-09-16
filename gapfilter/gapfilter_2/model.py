"""
GapFilter-2 网络：

1. 对 concat 的 local|contextual 特征做门控融合；
2. 单点编码器 f_θ(x) -> 表达嵌入；
3. 定义 Δy_ij = f_θ(x_i) - f_θ(x_j)，天然反对称，无需 antisymmetry loss；
4. 损失：MAE(y_i - y_j, Δy_ij)。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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


class GatedFusionEncoder(nn.Module):
    """门控融合 local/contextual，再映射到 gene_dim 表达嵌入。"""

    def __init__(
        self,
        feature_dim: int,
        gene_dim: int,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        if feature_dim % 2 != 0:
            raise ValueError(f"feature_dim={feature_dim} 必须为偶数（local|contextual）")
        self.branch_dim = feature_dim // 2

        self.gate = nn.Sequential(
            nn.Linear(self.branch_dim * 2, self.branch_dim),
            nn.Sigmoid(),
        )

        layers: list[nn.Module] = []
        prev = self.branch_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, int(h)), nn.GELU(), nn.Dropout(dropout)])
            prev = int(h)
        layers.append(nn.Linear(prev, gene_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = x[..., : self.branch_dim]
        contextual = x[..., self.branch_dim :]
        gate = self.gate(torch.cat([local, contextual], dim=-1))
        fused = gate * local + (1.0 - gate) * contextual
        return self.mlp(fused)


class GapFilterModel(pl.LightningModule):
    """Δy_ij = f_θ(x_i) - f_θ(x_j)。

    batch 字段：x1, x2, y1, y2（原始 log1p_cpm）。
    """

    def __init__(self, config: Any = None):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        feature_dim = int(_cfg_get(config, "feature_dim", 3072))
        gene_dim = int(_cfg_get(config, "gene_dim", 200))
        hidden_dims = tuple(_cfg_get(config, "hidden_dims", (1024, 512)))
        dropout = float(_cfg_get(config, "dropout", 0.1))
        self.lr = float(_cfg_get(config, "lr", 1e-3))
        self.weight_decay = float(_cfg_get(config, "weight_decay", 1e-4))

        self.encoder = GatedFusionEncoder(
            feature_dim=feature_dim,
            gene_dim=gene_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Δy_12 = f(x1) - f(x2)。"""
        return self.encode(x1) - self.encode(x2)

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
    ) -> dict:
        y_pred = self(x1, x2)
        y_true = y1 - y2
        loss = F.l1_loss(y_pred, y_true)
        return {"loss": loss, "y_pred": y_pred}

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
            "y_true": batch["y1"] - batch["y2"],
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
