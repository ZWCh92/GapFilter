"""
GapFilter-3：形态学引导的基因表达注意力生成。

Q、K：H&E feature 经线性映射
V：邻居表达值经线性映射到 d_model，注意力聚合后再映回 gene_dim
输出：ŷ = Out(softmax(QKᵀ/√d) V)

双层优化：
1. L_fit：用 train 邻居重建 train query；
2. L_cycle：先预测 test，再用预测的 test 回写 train，逼近真实 train 表达。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from gapfilter.gapfilter_3.dataset import CycleGraph


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class MorphologyGuidedAttention(nn.Module):
    """多头注意力：Q/K←形态学，V←表达（均经线性映射）。"""

    def __init__(
        self,
        feature_dim: int,
        gene_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.gene_dim = int(gene_dim)

        self.q_proj = nn.Linear(feature_dim, self.d_model)
        self.k_proj = nn.Linear(feature_dim, self.d_model)
        self.v_proj = nn.Linear(gene_dim, self.d_model)
        self.out_proj = nn.Linear(self.d_model, gene_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query_x: torch.Tensor,
        nbr_x: torch.Tensor,
        nbr_y: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query_x: (B, F)
        nbr_x:   (B, K, F)
        nbr_y:   (B, K, G)
        nbr_mask:(B, K) True=有效
        return:  (B, G)
        """
        bsz, k, _ = nbr_x.shape
        q = self.q_proj(query_x)  # (B, D)
        key = self.k_proj(nbr_x)  # (B, K, D)
        val = self.v_proj(nbr_y)  # (B, K, D)

        q = q.view(bsz, self.n_heads, self.head_dim)  # (B, H, Dh)
        key = key.view(bsz, k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        val = val.view(bsz, k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # key/val: (B, H, K, Dh)

        scale = self.head_dim ** -0.5
        scores = torch.einsum("bhd,bhkd->bhk", q, key) * scale  # (B, H, K)

        if nbr_mask is not None:
            mask = nbr_mask.unsqueeze(1)  # (B, 1, K)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        # 全被 mask 时 softmax 出 nan，置 0
        if nbr_mask is not None:
            all_invalid = ~nbr_mask.any(dim=-1)  # (B,)
            attn = torch.where(
                all_invalid.view(bsz, 1, 1),
                torch.zeros_like(attn),
                attn,
            )
        attn = self.attn_drop(attn)

        # (B, H, K) x (B, H, K, Dh) -> (B, H, Dh) -> (B, D)
        out = torch.einsum("bhk,bhkd->bhd", attn, val)
        out = out.reshape(bsz, self.d_model)
        return self.out_proj(out)


class GapFilterModel(pl.LightningModule):
    """双层损失：L = L_fit + cycle_weight * L_cycle。"""

    def __init__(self, config: Any = None):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        feature_dim = int(_cfg_get(config, "feature_dim", 3072))
        gene_dim = int(_cfg_get(config, "gene_dim", 200))
        d_model = int(_cfg_get(config, "d_model", 256))
        n_heads = int(_cfg_get(config, "n_heads", 4))
        dropout = float(_cfg_get(config, "dropout", 0.1))

        self.cycle_weight = float(_cfg_get(config, "cycle_weight", 1.0))
        self.cycle_chunk = int(_cfg_get(config, "cycle_chunk", 256))
        self.lr = float(_cfg_get(config, "lr", 1e-3))
        self.weight_decay = float(_cfg_get(config, "weight_decay", 1e-4))

        self.attn = MorphologyGuidedAttention(
            feature_dim=feature_dim,
            gene_dim=gene_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # 由 train.py 注入
        self._train_features: Optional[torch.Tensor] = None
        self._train_expr: Optional[torch.Tensor] = None
        self._cycle: Optional[CycleGraph] = None

    def set_context(
        self,
        train_features: torch.Tensor,
        train_expr: torch.Tensor,
        cycle_graph: CycleGraph,
    ) -> None:
        """注册 train 张量与 cycle 图（不作为 buffer，避免 ckpt 过大）。"""
        self._train_features = train_features
        self._train_expr = train_expr
        self._cycle = cycle_graph

    def forward(
        self,
        query_x: torch.Tensor,
        nbr_x: torch.Tensor,
        nbr_y: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attn(query_x, nbr_x, nbr_y, nbr_mask)

    def _gather_neighbors(
        self,
        features: torch.Tensor,
        expr: Optional[torch.Tensor],
        nbr_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """nbr_idx: (B, K)，-1 为无效。"""
        mask = nbr_idx >= 0
        safe = nbr_idx.clamp(min=0)
        nbr_x = features[safe]
        # 无效位置清零特征，避免泄漏到 QK（已有 mask）
        nbr_x = nbr_x * mask.unsqueeze(-1).to(nbr_x.dtype)
        nbr_y = None
        if expr is not None:
            nbr_y = expr[safe] * mask.unsqueeze(-1).to(expr.dtype)
        return nbr_x, nbr_y, mask

    def compute_cycle_loss(self) -> torch.Tensor:
        """ŷ_test ← train；再 ŷ_train ← ŷ_test；对齐真实 train。"""
        if (
            self._cycle is None
            or self._train_features is None
            or self._train_expr is None
            or len(self._cycle.cycle_train_idx) == 0
        ):
            return torch.zeros((), device=self.device)

        cycle = self._cycle
        train_x = self._train_features.to(self.device)
        train_y = self._train_expr.to(self.device)
        test_x = torch.as_tensor(cycle.test_features, device=self.device)

        # ---- 1) 预测全部 test ----
        t2tr = torch.as_tensor(cycle.test_to_train_idx, device=self.device)
        pred_test_parts = []
        n_test = test_x.shape[0]
        for start in range(0, n_test, self.cycle_chunk):
            end = min(start + self.cycle_chunk, n_test)
            q = test_x[start:end]
            nbr_x, nbr_y, mask = self._gather_neighbors(
                train_x, train_y, t2tr[start:end]
            )
            pred_test_parts.append(self(q, nbr_x, nbr_y, mask))
        pred_test = torch.cat(pred_test_parts, dim=0)  # (T, G)

        # ---- 2) 用预测 test 回写 cycle train ----
        c_idx = torch.as_tensor(cycle.cycle_train_idx, device=self.device, dtype=torch.long)
        tr2te = torch.as_tensor(cycle.train_to_test_idx, device=self.device)
        loss_acc = train_y.new_zeros(())
        n_valid = 0
        for start in range(0, c_idx.numel(), self.cycle_chunk):
            end = min(start + self.cycle_chunk, c_idx.numel())
            qi = c_idx[start:end]
            q = train_x[qi]
            nbr_test_idx = tr2te[qi]  # (B, K) -> test 局部
            nbr_x, _, mask = self._gather_neighbors(test_x, None, nbr_test_idx)
            # V 来自预测的 test 表达
            safe = nbr_test_idx.clamp(min=0)
            nbr_y = pred_test[safe] * mask.unsqueeze(-1).to(pred_test.dtype)
            pred_train = self(q, nbr_x, nbr_y, mask)
            # 至少有一个有效邻居的样本才计入
            row_valid = mask.any(dim=-1)
            if row_valid.any():
                loss_acc = loss_acc + F.l1_loss(
                    pred_train[row_valid], train_y[qi][row_valid]
                )
                n_valid += 1
        if n_valid == 0:
            return torch.zeros((), device=self.device)
        return loss_acc / n_valid

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        y_pred = self(
            batch["query_x"],
            batch["nbr_x"],
            batch["nbr_y"],
            batch["nbr_mask"],
        )
        loss_fit = F.l1_loss(y_pred, batch["query_y"])

        if stage == "train" and self.cycle_weight > 0:
            loss_cycle = self.compute_cycle_loss()
            loss = loss_fit + self.cycle_weight * loss_cycle
        else:
            loss_cycle = torch.zeros((), device=self.device)
            loss = loss_fit

        bs = batch["query_x"].size(0)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        self.log(
            f"{stage}/loss_fit",
            loss_fit,
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        if stage == "train":
            self.log(
                "train/loss_cycle",
                loss_cycle.detach(),
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
        return loss

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):
        return self._step(batch, "test")

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
