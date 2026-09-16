"""
加载 GapFilter-3 模型，对 test spot 做形态学引导注意力预测并评估。

对每个 test spot：以 K 个最近 train spot 为键值源，
ŷ_test = Attention(Q=x_test, K=x_train, V=y_train)。
多折 ensemble 时对预测取平均。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Union

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

from gapfilter.gapfilter_3.dataset import _infer_feature_key, knn_indices
from gapfilter.gapfilter_3.model import GapFilterModel


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _as_gene_names(gene_list) -> List[str]:
    if isinstance(gene_list, pd.DataFrame):
        return gene_list.iloc[:, 0].astype(str).tolist()
    if isinstance(gene_list, pd.Series):
        return gene_list.astype(str).tolist()
    return [str(g) for g in gene_list]


def evaluate(pred_X, test_X, gene_list, var_names):
    genes = _as_gene_names(gene_list)
    name_to_idx = {g: i for i, g in enumerate(map(str, var_names))}
    idxs = [name_to_idx[g] for g in genes if g in name_to_idx]
    if not idxs:
        raise ValueError("gene_list 与 var_names 无交集，无法评估")

    pred = np.asarray(pred_X, dtype=np.float64)[:, idxs]
    true = np.asarray(test_X, dtype=np.float64)[:, idxs]

    pccs, sccs = [], []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], true[:, i]
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            continue
        pccs.append(pearsonr(p, t)[0])
        sccs.append(spearmanr(p, t)[0])

    pcc = float(np.nanmean(pccs)) if pccs else float("nan")
    scc = float(np.nanmean(sccs)) if sccs else float("nan")
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return pcc, scc, rmse


def _resolve_sample_dirs(cfg: DictConfig) -> List[str]:
    samples = OmegaConf.select(cfg.dataset, "samples", default=None)
    if samples:
        return [
            os.path.join(cfg.dataset.processed_data_dir, str(s)) for s in samples
        ]
    return [cfg.dataset.processed_data_dir]


def _load_gene_names(sample_ckpt_dir: Path) -> np.ndarray:
    gene_path = sample_ckpt_dir / "gene_names.npz"
    if not gene_path.is_file():
        raise FileNotFoundError(f"未找到 gene_names.npz: {gene_path}")
    return np.asarray(np.load(gene_path, allow_pickle=True)["gene_names"], dtype=str)


def _find_ckpt_paths(sample_ckpt_dir: Path, fold: Union[str, int]) -> List[Path]:
    fold_dirs = sorted(sample_ckpt_dir.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"未找到 fold 目录: {sample_ckpt_dir}/fold_*")

    def _best_ckpt(d: Path) -> Path:
        cand = d / "best.ckpt"
        if cand.is_file():
            return cand
        ckpts = list(d.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"未找到 checkpoint: {d}")
        return ckpts[0]

    fold = str(fold)
    if fold == "ensemble":
        return [_best_ckpt(d) for d in fold_dirs]

    try:
        fold_id = int(fold)
    except ValueError as exc:
        raise ValueError(f"无效 fold={fold!r}，请使用 ensemble 或折编号") from exc

    target = sample_ckpt_dir / f"fold_{fold_id}"
    if not target.is_dir():
        raise FileNotFoundError(f"未找到折目录: {target}")
    return [_best_ckpt(target)]


def _load_models(
    ckpt_paths: Sequence[Path],
    feature_dim: int,
    gene_dim: int,
    cfg: DictConfig,
    device: torch.device,
) -> List[GapFilterModel]:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["feature_dim"] = feature_dim
    model_cfg["gene_dim"] = gene_dim
    models = []
    for ckpt in ckpt_paths:
        model = GapFilterModel.load_from_checkpoint(
            str(ckpt),
            config=model_cfg,
            map_location=device,
        )
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded checkpoint: {ckpt}")
    return models


@torch.inference_mode()
def predict_test_spots(
    models: Sequence[GapFilterModel],
    train_features: np.ndarray,
    train_expr: np.ndarray,
    test_features: np.ndarray,
    neighbor_idx: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """ŷ_test = Attn(Q=x_test, K=x_train_nn, V=y_train_nn)。"""
    n_test, k = neighbor_idx.shape
    gene_dim = train_expr.shape[1]
    pred_acc = np.zeros((n_test, gene_dim), dtype=np.float32)

    train_x = torch.from_numpy(train_features).to(device)
    train_y = torch.from_numpy(train_expr).to(device)
    test_x = torch.from_numpy(test_features).to(device)
    nbr_idx_t = torch.from_numpy(neighbor_idx.astype(np.int64)).to(device)

    for model in models:
        parts = []
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            q = test_x[start:end]
            idx = nbr_idx_t[start:end]
            mask = idx >= 0
            safe = idx.clamp(min=0)
            nbr_x = train_x[safe] * mask.unsqueeze(-1).to(train_x.dtype)
            nbr_y = train_y[safe] * mask.unsqueeze(-1).to(train_y.dtype)
            parts.append(model(q, nbr_x, nbr_y, mask).float().cpu().numpy())
        pred_acc += np.concatenate(parts, axis=0)
    pred_acc /= len(models)
    return pred_acc


def predict_one_sample(
    cfg: DictConfig,
    sample_dir: str,
    out_dir: Path,
    device: torch.device,
) -> dict:
    sample_name = Path(sample_dir).name
    ckpt_root = Path(cfg.model.ckpt_dir)
    if not ckpt_root.is_dir():
        raise FileNotFoundError(
            f"ckpt_dir 无效: {ckpt_root}。请传入训练输出目录，"
            f"例如 model.ckpt_dir=models/gapfilter_3/..."
        )

    sample_ckpt_dir = ckpt_root / sample_name
    if not sample_ckpt_dir.is_dir():
        if (ckpt_root / "gene_names.npz").is_file():
            sample_ckpt_dir = ckpt_root
        else:
            raise FileNotFoundError(
                f"未找到样本 checkpoint 目录: {ckpt_root / sample_name}"
            )

    gene_names = _load_gene_names(sample_ckpt_dir)

    h5ad_path = os.path.join(sample_dir, cfg.model.features_filename)
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f"未找到特征文件: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    feature_key = _infer_feature_key(
        adata, OmegaConf.select(cfg.model, "feature_key", default=None)
    )
    features = np.asarray(adata.obsm[feature_key], dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    expr = _to_dense(adata.layers["log1p_cpm"]).astype(np.float32)

    name_to_idx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    gene_idxs = np.asarray([name_to_idx[g] for g in gene_names if g in name_to_idx])
    if len(gene_idxs) != len(gene_names):
        missing = [g for g in gene_names if g not in name_to_idx]
        raise KeyError(f"gene_names 中有基因不在 adata: {missing[:5]}")

    train_mask = adata.obs["split"].astype(str).values == "train"
    test_mask = adata.obs["split"].astype(str).values == "test"
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError(f"{sample_name} 无 test spot")

    k = int(cfg.model.predict_n_neighbors)
    if len(train_idx) < k:
        raise ValueError(
            f"train spot 数 ({len(train_idx)}) < predict_n_neighbors ({k})"
        )

    max_distance = OmegaConf.select(cfg.model, "max_distance", default=None)
    neighbor_local, _ = knn_indices(
        coords[test_idx],
        coords[train_idx],
        n_neighbors=k,
        max_distance=max_distance,
        exclude_self=False,
    )

    train_features = features[train_idx]
    test_features = features[test_idx]
    train_expr_genes = expr[train_idx][:, gene_idxs]

    feature_dim = int(features.shape[1])
    gene_dim = int(len(gene_names))
    ckpt_paths = _find_ckpt_paths(
        sample_ckpt_dir, OmegaConf.select(cfg.model, "fold", default="ensemble")
    )
    models = _load_models(ckpt_paths, feature_dim, gene_dim, cfg, device)

    pred_genes = predict_test_spots(
        models=models,
        train_features=train_features,
        train_expr=train_expr_genes,
        test_features=test_features,
        neighbor_idx=neighbor_local,
        device=device,
        batch_size=int(cfg.model.batch_size),
    )

    test_X = expr[test_idx]
    pred_X = np.repeat(expr[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)
    pred_X[:, gene_idxs] = pred_genes

    general_svg = pd.read_csv(
        os.path.join(sample_dir, cfg.dataset.files.general_svg),
        sep="\t",
        header=None,
    )
    top_svg = pd.read_csv(
        os.path.join(sample_dir, cfg.dataset.files.top_svg),
        sep="\t",
        header=None,
    )

    general_pcc, general_scc, general_rmse = evaluate(
        pred_X, test_X, general_svg, adata.var_names
    )
    top_pcc, top_scc, top_rmse = evaluate(
        pred_X, test_X, top_svg, adata.var_names
    )
    print(
        f"[{sample_name}] General PCC: {general_pcc:.4f}, "
        f"SCC: {general_scc:.4f}, RMSE: {general_rmse:.4f}"
    )
    print(
        f"[{sample_name}] Top PCC: {top_pcc:.4f}, "
        f"SCC: {top_scc:.4f}, RMSE: {top_rmse:.4f}"
    )

    samples = OmegaConf.select(cfg.dataset, "samples", default=None)
    eval_results = {
        "sample_name": sample_name,
        "general_pcc": float(general_pcc),
        "general_scc": float(general_scc),
        "general_rmse": float(general_rmse),
        "top_pcc": float(top_pcc),
        "top_scc": float(top_scc),
        "top_rmse": float(top_rmse),
        "predict_n_neighbors": k,
        "n_models": len(models),
        "ckpt_dir": str(sample_ckpt_dir),
    }
    eval_save_name = (
        f"eval_results_{sample_name}.yaml" if samples else "eval_results.yaml"
    )
    with open(out_dir / eval_save_name, "w") as f:
        OmegaConf.save(config=OmegaConf.create(eval_results), f=f.name)

    pred_adata = adata.copy()
    layer = _to_dense(pred_adata.layers["log1p_cpm"]).astype(np.float32)
    layer[test_mask] = pred_X
    pred_adata.layers["log1p_cpm"] = layer
    save_name = f"pred_adata_{sample_name}.h5ad" if samples else "pred_adata.h5ad"
    pred_adata.write_h5ad(out_dir / save_name)
    print(f"[{sample_name}] saved -> {out_dir / save_name}")
    return eval_results


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cuda_id = int(OmegaConf.select(cfg, "cuda", default=0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample_dir in _resolve_sample_dirs(cfg):
        results.append(predict_one_sample(cfg, sample_dir, out_dir, device))

    with open(out_dir / "predict_summary.yaml", "w") as f:
        OmegaConf.save(config=OmegaConf.create(results), f=f.name)
    print(f"Prediction finished. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
