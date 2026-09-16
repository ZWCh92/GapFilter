"""
GapFilter-4：形态学引导 + 距离 bias + 低维残差修正。

相对 GapFilter-3 的主要改动：
1. 每 head 可学习距离 bias（softplus 保证非负；head0 固定为无距离）；
2. 基线 = 注意力加权邻居原始表达（凸组合）；
3. 可选低维残差：Δy = γ · U [s ⊙ tanh(Δz)]，U/s 来自 base 残差 PCA；
4. Morph encoder：输入 LN + MLP + LN，再投影 Q/K；
5. L_fit = L1 + λ_p * Pearson + λ_c * Cosine。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Union

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


def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Gene-wise Pearson：batch 维相关，返回 1 - mean(corr)。"""
    if pred.size(0) < 2:
        return pred.new_zeros(())
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * tgt_c).sum(dim=0)
    den = pred_c.norm(dim=0) * tgt_c.norm(dim=0)
    corr = num / den.clamp_min(eps)
    valid = den > eps
    if not valid.any():
        return pred.new_zeros(())
    return (1.0 - corr[valid]).mean()


def _cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Sample-wise cosine：基因维，返回 1 - mean(cos)。"""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=eps)).mean()


def compute_residual_pca(
    residuals: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对残差矩阵 R (N, G) 做 PCA，返回 U (G, r) 与尺度 s (r,)。

    s_j = σ_j = S_j / sqrt(N-1)，对应主成分方向上的典型残差幅度。
    """
    if residuals.ndim != 2:
        raise ValueError(f"residuals 应为 (N, G)，得到 shape={tuple(residuals.shape)}")
    n_samples, gene_dim = residuals.shape
    if n_samples < 2:
        raise ValueError("残差 PCA 至少需要 2 个样本")

    r = int(min(rank, gene_dim, n_samples - 1))
    if r < 1:
        raise ValueError(f"无效 residual_rank={rank}")

    r_center = residuals - residuals.mean(dim=0, keepdim=True)
    # full_matrices=False: Vh 为 (min(N,G), G)
    _u, svals, vh = torch.linalg.svd(r_center, full_matrices=False)
    u_basis = vh[:r].transpose(0, 1).contiguous()  # (G, r)
    scale = svals[:r] / math.sqrt(max(n_samples - 1, 1))
    scale = scale.clamp_min(1e-6)
    return u_basis, scale


class MorphEncoder(nn.Module):
    """轻量形态学编码器：LN → Linear → GELU → Linear → LN。"""

    def __init__(self, feature_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.mlp(self.in_norm(x)))


class MorphologyGuidedAttention(nn.Module):
    """多头注意力：Q/K←形态学(+距离 bias)，基线=attn@表达，可选低维残差。"""

    def __init__(
        self,
        feature_dim: int,
        gene_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        dist_lambda_init: float = -1.0,
        zero_dist_head: bool = True,
        use_dist_bias: bool = True,
        use_gated_residual: bool = True,
        residual_rank: int = 16,
        residual_gamma: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.gene_dim = int(gene_dim)
        self.use_dist_bias = bool(use_dist_bias)
        self.zero_dist_head = bool(zero_dist_head) and self.n_heads > 1 and self.use_dist_bias
        self.use_gated_residual = bool(use_gated_residual)
        self.residual_rank = int(residual_rank)
        self.residual_gamma = float(residual_gamma)

        self.morph = MorphEncoder(feature_dim, self.d_model, dropout=dropout)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)

        if self.use_gated_residual:
            if self.residual_rank < 1:
                raise ValueError("residual_rank 必须 >= 1")
            self.v_proj = nn.Linear(gene_dim, self.d_model)
            # Δz = f_res(q_h, ctx)，只预测 r 维系数
            self.delta_z_mlp = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.residual_rank),
            )
            # U: (G, r)，s: (r,)；由 residual PCA 写入，训练中冻结
            self.register_buffer(
                "residual_U",
                torch.zeros(self.gene_dim, self.residual_rank),
                persistent=True,
            )
            self.register_buffer(
                "residual_s",
                torch.ones(self.residual_rank),
                persistent=True,
            )
            self.register_buffer(
                "residual_basis_ready",
                torch.tensor(False),
                persistent=True,
            )
        else:
            self.v_proj = None
            self.delta_z_mlp = None

        self.attn_drop = nn.Dropout(dropout)

        if self.use_dist_bias:
            init = torch.full((self.n_heads,), float(dist_lambda_init))
            if self.zero_dist_head:
                init[0] = -10.0
            self.log_lambda_dist = nn.Parameter(init)
        else:
            self.register_parameter("log_lambda_dist", None)

    def set_residual_basis(
        self,
        u_basis: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """写入残差 PCA 基：U (G, r)，s (r,)。"""
        if not self.use_gated_residual:
            raise RuntimeError("use_gated_residual=False 时无需设置 residual basis")
        if u_basis.shape != (self.gene_dim, self.residual_rank):
            raise ValueError(
                f"U 期望 shape=({self.gene_dim}, {self.residual_rank})，"
                f"得到 {tuple(u_basis.shape)}"
            )
        if scale.shape != (self.residual_rank,):
            raise ValueError(
                f"s 期望 shape=({self.residual_rank},)，得到 {tuple(scale.shape)}"
            )
        self.residual_U.copy_(u_basis.to(dtype=self.residual_U.dtype))
        self.residual_s.copy_(scale.to(dtype=self.residual_s.dtype).clamp_min(1e-6))
        self.residual_basis_ready.fill_(True)

    def fit_residual_basis(self, residuals: torch.Tensor) -> dict:
        """对 (N, G) 残差做 PCA 并写入 U、s。"""
        u_basis, scale = compute_residual_pca(residuals.detach().float().cpu(), self.residual_rank)
        # 若 SVD 秩不足 residual_rank，右侧补零列（极少见）
        if u_basis.shape[1] < self.residual_rank:
            pad = self.residual_rank - u_basis.shape[1]
            u_basis = F.pad(u_basis, (0, pad))
            scale = F.pad(scale, (0, pad))
        self.set_residual_basis(u_basis.to(self.residual_U.device), scale.to(self.residual_s.device))
        explained = scale.pow(2)
        total = explained.sum().clamp_min(1e-12)
        return {
            "residual_rank": int(self.residual_rank),
            "n_samples": int(residuals.shape[0]),
            "scale_mean": float(scale.mean()),
            "explained_ratio_top": float((explained / total).sum()),
        }

    def _lambda_dist(self) -> torch.Tensor:
        lam = F.softplus(self.log_lambda_dist)  # (H,)
        if self.zero_dist_head:
            lam = torch.cat([lam.new_zeros(1), lam[1:]], dim=0)
        return lam

    def _normalize_dist(
        self,
        nbr_dist: torch.Tensor,
        nbr_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """按 query 内有效邻居均值归一化距离，无效位置置 0（随后被 mask）。"""
        dist = nbr_dist.clamp_min(0.0)
        if nbr_mask is None:
            scale = dist.mean(dim=-1, keepdim=True).clamp_min(1e-6)
            return dist / scale

        mask_f = nbr_mask.to(dist.dtype)
        denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        scale = (dist * mask_f).sum(dim=-1, keepdim=True) / denom
        scale = scale.clamp_min(1e-6)
        norm = dist / scale
        return norm * mask_f

    def forward(
        self,
        query_x: torch.Tensor,
        nbr_x: torch.Tensor,
        nbr_y: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
        nbr_dist: Optional[torch.Tensor] = None,
        return_base_only: bool = False,
    ) -> torch.Tensor:
        """
        query_x: (B, F)
        nbr_x:   (B, K, F)
        nbr_y:   (B, K, G)
        nbr_mask:(B, K) True=有效
        nbr_dist:(B, K) 空间距离
        return:  (B, G)
        """
        bsz, k, _ = nbr_x.shape

        q_h = self.morph(query_x)  # (B, D)
        nbr_h = self.morph(nbr_x)  # (B, K, D)

        q = self.q_proj(q_h)
        key = self.k_proj(nbr_h)

        q = q.view(bsz, self.n_heads, self.head_dim)
        key = key.view(bsz, k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        scores = torch.einsum("bhd,bhkd->bhk", q, key) * scale  # (B, H, K)

        if self.use_dist_bias and nbr_dist is not None:
            norm_dist = self._normalize_dist(nbr_dist, nbr_mask)  # (B, K)
            lam = self._lambda_dist()  # (H,)
            scores = scores - lam.view(1, self.n_heads, 1) * norm_dist.square().unsqueeze(1)

        if nbr_mask is not None:
            mask = nbr_mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        if nbr_mask is not None:
            all_invalid = ~nbr_mask.any(dim=-1)
            attn = torch.where(
                all_invalid.view(bsz, 1, 1),
                torch.zeros_like(attn),
                attn,
            )

        # Dropout 后再归一化：保证训练时 base 仍是邻居凸组合
        attn = self.attn_drop(attn)
        if nbr_mask is not None:
            attn = attn * nbr_mask.unsqueeze(1).to(attn.dtype)
        head_sum = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(
            head_sum > 0,
            attn / head_sum.clamp_min(1e-12),
            torch.zeros_like(attn),
        )

        attn_gene = attn.mean(dim=1)  # (B, K)
        gene_sum = attn_gene.sum(dim=-1, keepdim=True)
        attn_gene = torch.where(
            gene_sum > 0,
            attn_gene / gene_sum.clamp_min(1e-12),
            attn_gene,
        )
        base = torch.einsum("bk,bkg->bg", attn_gene, nbr_y)

        use_res = (
            self.use_gated_residual
            and (not return_base_only)
            and bool(self.residual_basis_ready.item())
        )
        if not use_res:
            return base

        # ---- 低维残差：Δy = γ · U [s ⊙ tanh(Δz)] ----
        val = self.v_proj(nbr_y)
        val = val.view(bsz, k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        ctx = torch.einsum("bhk,bhkd->bhd", attn, val).reshape(bsz, self.d_model)
        # 残差头不回传打乱 base 注意力：对 ctx/q_h 保持梯度，但对 base 本身 stop-grad
        delta_z = self.delta_z_mlp(torch.cat([q_h, ctx], dim=-1))  # (B, r)
        coeff = self.residual_s.unsqueeze(0) * torch.tanh(delta_z)  # (B, r)
        delta_y = self.residual_gamma * (coeff @ self.residual_U.transpose(0, 1))
        return base + delta_y


class GapFilterModel(pl.LightningModule):
    """L = L1 + λ_p · Pearson + λ_c · Cosine。"""

    def __init__(self, config: Any = None):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        feature_dim = int(_cfg_get(config, "feature_dim", 3072))
        gene_dim = int(_cfg_get(config, "gene_dim", 200))
        d_model = int(_cfg_get(config, "d_model", 256))
        n_heads = int(_cfg_get(config, "n_heads", 4))
        dropout = float(_cfg_get(config, "dropout", 0.1))

        self.pearson_weight = float(_cfg_get(config, "pearson_weight", 0.5))
        self.cosine_weight = float(_cfg_get(config, "cosine_weight", 0.1))

        self.lr = float(_cfg_get(config, "lr", 1e-3))
        self.weight_decay = float(_cfg_get(config, "weight_decay", 1e-4))

        self.attn = MorphologyGuidedAttention(
            feature_dim=feature_dim,
            gene_dim=gene_dim,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            dist_lambda_init=float(_cfg_get(config, "dist_lambda_init", -1.0)),
            zero_dist_head=bool(_cfg_get(config, "zero_dist_head", True)),
            use_dist_bias=bool(_cfg_get(config, "use_dist_bias", True)),
            use_gated_residual=bool(_cfg_get(config, "use_gated_residual", True)),
            residual_rank=int(_cfg_get(config, "residual_rank", 16)),
            residual_gamma=float(_cfg_get(config, "residual_gamma", 0.1)),
        )

    def forward(
        self,
        query_x: torch.Tensor,
        nbr_x: torch.Tensor,
        nbr_y: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
        nbr_dist: Optional[torch.Tensor] = None,
        return_base_only: bool = False,
    ) -> torch.Tensor:
        return self.attn(
            query_x, nbr_x, nbr_y, nbr_mask, nbr_dist, return_base_only=return_base_only
        )

    def fit_residual_basis(self, residuals: torch.Tensor) -> dict:
        """代理到 attention 模块的残差 PCA。"""
        return self.attn.fit_residual_basis(residuals)

    @torch.no_grad()
    def collect_base_residuals(
        self,
        dataloader,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        """在 dataloader 上收集 y_true - base，用于残差 PCA。"""
        was_training = self.training
        self.eval()
        parts = []
        for batch in dataloader:
            if device is not None:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
                }
            base = self(
                batch["query_x"],
                batch["nbr_x"],
                batch["nbr_y"],
                batch["nbr_mask"],
                batch.get("nbr_dist"),
                return_base_only=True,
            )
            parts.append((batch["query_y"] - base).detach().cpu())
        if was_training:
            self.train()
        if not parts:
            raise RuntimeError("collect_base_residuals: dataloader 为空")
        return torch.cat(parts, dim=0)

    def _fit_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss_l1 = F.l1_loss(y_pred, y_true)
        loss_pearson = _pearson_loss(y_pred, y_true)
        loss_cosine = _cosine_loss(y_pred, y_true)
        loss = (
            loss_l1
            + self.pearson_weight * loss_pearson
            + self.cosine_weight * loss_cosine
        )
        return loss, {
            "l1": loss_l1,
            "pearson": loss_pearson,
            "cosine": loss_cosine,
        }

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        nbr_dist = batch.get("nbr_dist")
        y_pred = self(
            batch["query_x"],
            batch["nbr_x"],
            batch["nbr_y"],
            batch["nbr_mask"],
            nbr_dist,
        )
        loss, fit_parts = self._fit_loss(y_pred, batch["query_y"])

        bs = batch["query_x"].size(0)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        self.log(f"{stage}/loss_l1", fit_parts["l1"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            f"{stage}/loss_pearson",
            fit_parts["pearson"],
            on_step=False,
            on_epoch=True,
            batch_size=bs,
        )
        self.log(
            f"{stage}/loss_cosine",
            fit_parts["cosine"],
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
