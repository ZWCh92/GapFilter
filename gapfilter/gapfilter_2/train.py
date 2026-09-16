"""
五折交叉验证训练（按 spot 划分，避免 pair 泄漏）：
读取 cfg -> Dataset -> 按 spot KFold 构图 -> Trainer -> 保存模型
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from gapfilter.gapfilter_2.dataset import GapFilterPairDataset
from gapfilter.gapfilter_2.model import GapFilterModel


def _resolve_sample_dirs(cfg: DictConfig) -> List[str]:
    samples = OmegaConf.select(cfg.dataset, "samples", default=None)
    if samples:
        return [
            os.path.join(cfg.dataset.processed_data_dir, str(s)) for s in samples
        ]
    return [cfg.dataset.processed_data_dir]


def _resolve_gene_list(cfg: DictConfig, sample_dir: str) -> Optional[str]:
    key = OmegaConf.select(cfg.model, "gene_list", default=None)
    if key is None:
        return None
    key = str(key)
    if key in ("null", "None", "all"):
        return None
    if key == "top_svg":
        return os.path.join(sample_dir, cfg.dataset.files.top_svg)
    if key == "general_svg":
        return os.path.join(sample_dir, cfg.dataset.files.general_svg)
    path = Path(key)
    if path.is_file():
        return str(path)
    raise ValueError(
        f"无法解析 gene_list={key!r}，请使用 top_svg / general_svg / null / 文件路径"
    )


def _build_model_config(cfg: DictConfig, feature_dim: int, gene_dim: int) -> dict:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["feature_dim"] = (
        int(model_cfg["feature_dim"])
        if model_cfg.get("feature_dim") is not None
        else feature_dim
    )
    model_cfg["gene_dim"] = (
        int(model_cfg["gene_dim"])
        if model_cfg.get("gene_dim") is not None
        else gene_dim
    )
    return model_cfg


def _make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def train_one_sample(cfg: DictConfig, sample_dir: str, out_dir: Path) -> dict:
    sample_name = Path(sample_dir).name
    h5ad_path = os.path.join(sample_dir, cfg.model.features_filename)
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"未找到特征文件: {h5ad_path}")

    gene_list = _resolve_gene_list(cfg, sample_dir)
    dataset = GapFilterPairDataset(
        h5ad_path=h5ad_path,
        feature_key=OmegaConf.select(cfg.model, "feature_key", default=None),
        gene_list=gene_list,
        n_neighbors=int(cfg.model.n_neighbors),
        max_distance=OmegaConf.select(cfg.model, "max_distance", default=None),
        bidirectional=bool(cfg.model.bidirectional),
        split="train",
    )

    feature_dim = int(dataset.features.shape[1])
    gene_dim = int(len(dataset.gene_names))
    model_cfg = _build_model_config(cfg, feature_dim, gene_dim)

    sample_out = out_dir / sample_name
    sample_out.mkdir(parents=True, exist_ok=True)
    dataset.save_gene_names(sample_out / "gene_names.npz")

    n_folds = int(cfg.model.n_folds)
    seed = int(cfg.model.seed)
    # 按 spot 划分，避免同一 spot 同时出现在 train/val 的不同 pair 中
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    spot_indices = np.arange(dataset.n_spots)

    fold_metrics = []
    cuda_id = int(OmegaConf.select(cfg, "cuda", default=0))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [cuda_id] if accelerator == "gpu" else 1

    for fold, (train_spots, val_spots) in enumerate(kf.split(spot_indices)):
        print(f"\n===== [{sample_name}] Fold {fold + 1}/{n_folds} (spot CV) =====")
        pl.seed_everything(seed + fold, workers=True)

        train_ds = dataset.subset_pairs(train_spots)
        val_ds = dataset.subset_pairs(val_spots)
        print(
            f"[{sample_name}] fold={fold}: "
            f"train_spots={len(train_spots)}, val_spots={len(val_spots)}, "
            f"train_pairs={len(train_ds)}, val_pairs={len(val_ds)}"
        )

        train_loader = _make_loader(
            train_ds,
            batch_size=int(cfg.model.batch_size),
            num_workers=int(cfg.model.num_workers),
            shuffle=True,
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=int(cfg.model.batch_size),
            num_workers=int(cfg.model.num_workers),
            shuffle=False,
        )

        model = GapFilterModel(model_cfg)
        fold_dir = sample_out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(fold_dir),
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        early_cb = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=int(cfg.model.patience),
            verbose=True,
        )

        trainer = pl.Trainer(
            default_root_dir=str(fold_dir),
            max_epochs=int(cfg.model.max_epochs),
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=float(cfg.model.gradient_clip_val),
            callbacks=[ckpt_cb, early_cb],
            enable_checkpointing=True,
            log_every_n_steps=10,
            deterministic=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_score = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None
            else float("nan")
        )
        fold_info = {
            "fold": fold,
            "best_val_loss": best_score,
            "best_model_path": ckpt_cb.best_model_path,
            "n_train_spots": int(len(train_spots)),
            "n_val_spots": int(len(val_spots)),
            "n_train_pairs": int(len(train_ds)),
            "n_val_pairs": int(len(val_ds)),
        }
        fold_metrics.append(fold_info)
        with open(fold_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(fold_info, f, sort_keys=False)
        print(
            f"[{sample_name}] fold={fold} best_val_loss={best_score:.6f} "
            f"ckpt={ckpt_cb.best_model_path}"
        )

    summary = {
        "sample": sample_name,
        "feature_dim": feature_dim,
        "gene_dim": gene_dim,
        "n_spots": int(dataset.n_spots),
        "n_genes": int(gene_dim),
        "gene_list": gene_list,
        "cv_split": "spot",
        "folds": fold_metrics,
        "mean_best_val_loss": float(
            np.nanmean([m["best_val_loss"] for m in fold_metrics])
        ),
    }
    with open(sample_out / "cv_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(
        f"[{sample_name}] CV mean best val/loss = {summary['mean_best_val_loss']:.6f}"
    )
    return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(int(cfg.model.seed), workers=True)

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = _resolve_sample_dirs(cfg)
    summaries = []
    for sample_dir in sample_dirs:
        summaries.append(train_one_sample(cfg, sample_dir, out_dir))

    with open(out_dir / "train_summary.yaml", "w") as f:
        yaml.safe_dump(summaries, f, sort_keys=False)
    print(f"Training finished. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
