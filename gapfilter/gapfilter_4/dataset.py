"""
GapFilter-4 Dataset：形态学引导注意力所需的近邻图（含距离）。

每个样本 = 一个 query spot + 其 K 个空间近邻（不含自身）：
- query / neighbor 的 H&E feature（供 Q、K）
- neighbor 的表达（供 V / gene-space 聚合）
- neighbor 的空间距离（供距离 bias）
- query 的表达（监督目标，仅 train/val 有标签）

不做 min-max；表达直接使用 log1p_cpm。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


PathLike = Union[str, Path]


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _read_gene_list(gene_list: Union[PathLike, Sequence[str], None]) -> Optional[List[str]]:
    if gene_list is None:
        return None
    if isinstance(gene_list, (str, Path)):
        path = Path(gene_list)
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(g) for g in gene_list]


def _infer_feature_key(adata: sc.AnnData, feature_key: Optional[str]) -> str:
    if feature_key is not None:
        if feature_key not in adata.obsm:
            raise KeyError(f"adata.obsm 中不存在 feature_key={feature_key}")
        return feature_key
    candidates = [k for k in adata.obsm.keys() if str(k).endswith("_features")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"发现多个特征键 {candidates}，请显式指定 feature_key")
    raise KeyError("adata.obsm 中未找到 '*_features'，请指定 feature_key")


def knn_indices(
    query_coords: np.ndarray,
    ref_coords: np.ndarray,
    n_neighbors: int,
    max_distance: Optional[float] = None,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (indices, distances)，shape (n_query, k)。

    exclude_self=True 时，在同一坐标系下跳过距离为 0 的自身（取 k+1 再丢弃）。
    若某 query 有效邻居不足 k，用 -1 填充 index，距离为 +inf。
    """
    n_query = query_coords.shape[0]
    n_ref = ref_coords.shape[0]
    if n_ref == 0 or n_query == 0:
        k = n_neighbors
        idx = -np.ones((n_query, k), dtype=np.int64)
        dist = np.full((n_query, k), np.inf, dtype=np.float64)
        return idx, dist

    k_search = min(n_ref, n_neighbors + (1 if exclude_self else 0))
    nn = NearestNeighbors(n_neighbors=k_search, algorithm="auto")
    nn.fit(ref_coords)
    distances, indices = nn.kneighbors(query_coords)

    out_idx = -np.ones((n_query, n_neighbors), dtype=np.int64)
    out_dist = np.full((n_query, n_neighbors), np.inf, dtype=np.float64)

    for i in range(n_query):
        cols = []
        for rank in range(k_search):
            j = int(indices[i, rank])
            d = float(distances[i, rank])
            if exclude_self and d <= 1e-12:
                continue
            if max_distance is not None and d > max_distance:
                continue
            cols.append((j, d))
            if len(cols) >= n_neighbors:
                break
        for t, (j, d) in enumerate(cols[:n_neighbors]):
            out_idx[i, t] = j
            out_dist[i, t] = d
    return out_idx, out_dist


class GapFilterAttnDataset(Dataset):
    """train spot 上的拟合样本：query + train 近邻。"""

    def __init__(
        self,
        h5ad_path: PathLike,
        feature_key: Optional[str] = None,
        gene_list: Union[PathLike, Sequence[str], None] = None,
        n_neighbors: int = 6,
        max_distance: Optional[float] = None,
    ):
        super().__init__()
        self.h5ad_path = Path(h5ad_path)
        self.n_neighbors = int(n_neighbors)
        self.max_distance = max_distance

        if not self.h5ad_path.is_file():
            raise FileNotFoundError(f"未找到特征文件: {self.h5ad_path}")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors 必须 >= 1")

        adata = sc.read_h5ad(self.h5ad_path)
        if "split" not in adata.obs:
            raise KeyError("adata.obs 缺少 'split' 列")
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm 缺少 'spatial'")
        if "log1p_cpm" not in adata.layers:
            raise KeyError("adata.layers 缺少 'log1p_cpm'")

        self.feature_key = _infer_feature_key(adata, feature_key)
        genes = _read_gene_list(gene_list)
        if genes is None:
            gene_mask = np.ones(adata.n_vars, dtype=bool)
            self.gene_names = np.asarray(adata.var_names, dtype=str)
        else:
            var_set = set(map(str, adata.var_names))
            keep = [g for g in genes if g in var_set]
            missing = len(genes) - len(keep)
            if missing:
                print(
                    f"[GapFilterAttnDataset] gene_list 中有 {missing} 个基因不在 adata 中，已忽略"
                )
            if not keep:
                raise ValueError("gene_list 与 adata.var_names 无交集")
            gene_mask = adata.var_names.isin(keep)
            self.gene_names = np.asarray(adata.var_names[gene_mask], dtype=str)

        split = adata.obs["split"].astype(str).values
        train_mask = split == "train"
        test_mask = split == "test"
        if train_mask.sum() < self.n_neighbors + 1:
            raise ValueError(
                f"train spot 数 ({train_mask.sum()}) 不足以取 n_neighbors={self.n_neighbors}"
            )

        features = np.asarray(adata.obsm[self.feature_key], dtype=np.float32)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        expr = _to_dense(adata.layers["log1p_cpm"][:, gene_mask]).astype(np.float32)

        self.train_idx_global = np.where(train_mask)[0]
        self.test_idx_global = np.where(test_mask)[0]

        self.train_features = features[self.train_idx_global]
        self.train_coords = coords[self.train_idx_global]
        self.train_expr = expr[self.train_idx_global]
        self.train_ids = np.asarray(adata.obs_names[train_mask], dtype=str)

        self.test_features = features[self.test_idx_global]
        self.test_coords = coords[self.test_idx_global]
        self.test_ids = np.asarray(adata.obs_names[test_mask], dtype=str)

        self.n_train = len(self.train_ids)
        self.n_test = len(self.test_ids)
        self.feature_dim = int(self.train_features.shape[1])
        self.gene_dim = int(self.train_expr.shape[1])

        print(
            f"[GapFilterAttnDataset] train={self.n_train}, test={self.n_test}, "
            f"genes={self.gene_dim}, feature_dim={self.feature_dim}, "
            f"n_neighbors={self.n_neighbors}, feature_key={self.feature_key}"
        )

    def make_fit_view(
        self,
        fit_spot_indices: np.ndarray,
        query_spot_indices: Optional[np.ndarray] = None,
    ) -> "AttnFitView":
        """在 fit_spot 池内为 query 构图。"""
        fit_spot_indices = np.asarray(fit_spot_indices, dtype=np.int64)
        if query_spot_indices is None:
            query_spot_indices = fit_spot_indices
        else:
            query_spot_indices = np.asarray(query_spot_indices, dtype=np.int64)

        pool_coords = self.train_coords[fit_spot_indices]
        q_coords = self.train_coords[query_spot_indices]
        same = (
            len(fit_spot_indices) == len(query_spot_indices)
            and np.array_equal(fit_spot_indices, query_spot_indices)
        )
        local_nbr, local_dist = knn_indices(
            q_coords,
            pool_coords,
            n_neighbors=self.n_neighbors,
            max_distance=self.max_distance,
            exclude_self=same,
        )
        mapped = np.full_like(local_nbr, -1)
        valid = local_nbr >= 0
        mapped[valid] = fit_spot_indices[local_nbr[valid]]

        return AttnFitView(
            parent=self,
            query_indices=query_spot_indices.copy(),
            fit_nbr_idx=mapped,
            fit_nbr_dist=local_dist,
        )

    def save_gene_names(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, gene_names=self.gene_names)


class AttnFitView(Dataset):
    """某一折上的拟合样本视图（query + fit-pool 近邻）。"""

    def __init__(
        self,
        parent: GapFilterAttnDataset,
        query_indices: np.ndarray,
        fit_nbr_idx: np.ndarray,
        fit_nbr_dist: np.ndarray,
    ):
        self.parent = parent
        self.query_indices = query_indices
        self.fit_nbr_idx = fit_nbr_idx
        self.fit_nbr_dist = fit_nbr_dist

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, index: int):
        q = int(self.query_indices[index])
        nbr = self.fit_nbr_idx[index]
        dist = self.fit_nbr_dist[index]
        mask = nbr >= 0
        safe_nbr = nbr.copy()
        safe_nbr[~mask] = q

        nbr_x = self.parent.train_features[safe_nbr]
        nbr_y = self.parent.train_expr[safe_nbr].copy()
        nbr_y[~mask] = 0.0
        nbr_dist = dist.astype(np.float32).copy()
        nbr_dist[~mask] = 0.0

        return {
            "query_x": torch.from_numpy(self.parent.train_features[q]),
            "query_y": torch.from_numpy(self.parent.train_expr[q]),
            "nbr_x": torch.from_numpy(nbr_x),
            "nbr_y": torch.from_numpy(nbr_y),
            "nbr_dist": torch.from_numpy(nbr_dist),
            "nbr_mask": torch.from_numpy(mask.astype(np.bool_)),
            "query_idx": q,
        }
