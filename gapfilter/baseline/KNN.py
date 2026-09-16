"""
对于gap区域的spot，直接使用最近的k个spot进行插值得到缺失值, k参数可调整
"""

import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _as_gene_names(gene_list):
    if isinstance(gene_list, pd.DataFrame):
        return gene_list.iloc[:, 0].astype(str).tolist()
    if isinstance(gene_list, pd.Series):
        return gene_list.astype(str).tolist()
    return [str(g) for g in gene_list]


def load_data(expr_path, general_svg_path, top_svg_path):
    adata = sc.read_h5ad(expr_path)
    general_svg = pd.read_csv(general_svg_path, sep="\t", header=None)
    top_svg = pd.read_csv(top_svg_path, sep="\t", header=None)
    return adata, general_svg, top_svg


def evaluate(pred_X, test_X, gene_list, var_names):
    """在 gene_list 子集上评估预测结果。

    对每个基因在 test spot 上计算 PCC / SCC，再取平均；RMSE 在子集整体上计算。
    """
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


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    samples = OmegaConf.select(cfg.dataset, "samples", default=None)
    if samples:
        sample_dirs = [os.path.join(cfg.dataset.processed_data_dir, str(s)) for s in samples]
    else:
        sample_dirs = [cfg.dataset.processed_data_dir]

    k = cfg.model.top_k if "top_k" in cfg.model else 6
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in sample_dirs:
        sample_name = Path(dataset_dir).name
        expr_path = os.path.join(dataset_dir, cfg.dataset.files.expression)
        general_svg_path = os.path.join(dataset_dir, cfg.dataset.files.general_svg)
        top_svg_path = os.path.join(dataset_dir, cfg.dataset.files.top_svg)

        adata, general_svg, top_svg = load_data(
            expr_path,  general_svg_path, top_svg_path
        )

        train_mask = adata.obs["split"].values == "train"
        test_mask = adata.obs["split"].values == "test"
        train_adata = adata[train_mask]
        test_adata = adata[test_mask]

        train_locs = np.asarray(train_adata.obsm["spatial"])
        test_locs = np.asarray(test_adata.obsm["spatial"])
        train_X = _to_dense(train_adata.layers["log1p_cpm"])
        test_X = _to_dense(test_adata.layers["log1p_cpm"])

        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(train_locs, train_X)
        pred_X = knn.predict(test_locs)

        general_pcc, general_scc, general_rmse = evaluate(
            pred_X, test_X, general_svg, adata.var_names
        )
        top_pcc, top_scc, top_rmse = evaluate(
            pred_X, test_X, top_svg, adata.var_names
        )

        print(f"[{sample_name}] General PCC: {general_pcc:.4f}, SCC: {general_scc:.4f}, RMSE: {general_rmse:.4f}")
        print(f"[{sample_name}] Top PCC: {top_pcc:.4f}, SCC: {top_scc:.4f}, RMSE: {top_rmse:.4f}")

        # 将评估结果也保存到out_dir
        eval_results = {
            "sample_name": sample_name,
            "general_pcc": float(general_pcc),
            "general_scc": float(general_scc),
            "general_rmse": float(general_rmse),
            "top_pcc": float(top_pcc),
            "top_scc": float(top_scc),
            "top_rmse": float(top_rmse)
        }
        eval_save_name = f"eval_results_{sample_name}.yaml" if samples else "eval_results.yaml"
        eval_save_path = out_dir / eval_save_name
        with open(eval_save_path, "w") as f:
            OmegaConf.save(config=OmegaConf.create(eval_results), f=f.name)
 

        # 预测在 log1p_cpm 尺度，写回对应 layer（AnnData 视图不能直接赋值）
        pred_adata = adata.copy()
        layer = _to_dense(pred_adata.layers["log1p_cpm"])
        layer[test_mask] = pred_X
        pred_adata.layers["log1p_cpm"] = layer

        save_name = f"pred_adata_{sample_name}.h5ad" if samples else "pred_adata.h5ad"
        pred_adata.write_h5ad(out_dir / save_name)


if __name__ == "__main__":
    main()
