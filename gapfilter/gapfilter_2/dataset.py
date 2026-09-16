"""
从 processed 目录加载 expr_with_HE_features.h5ad，构建近邻配对 Dataset。

- 表达直接使用 layers['log1p_cpm']，不做 min-max；
- n_neighbors 不含自身；
- 支持按 spot 子集构图，便于 spot 级交叉验证（避免 pair 泄漏）。
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


class _PairView(Dataset):
    """基于父 Dataset 的 pair 子集视图。"""

    def __init__(self, parent: "GapFilterPairDataset", pairs: List[tuple]):
        self.parent = parent
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j = self.pairs[index]
        return {
            "x1": torch.from_numpy(self.parent.features[i]),
            "x2": torch.from_numpy(self.parent.features[j]),
            "y1": torch.from_numpy(self.parent.expr[i]),
            "y2": torch.from_numpy(self.parent.expr[j]),
            "idx1": i,
            "idx2": j,
        }


class GapFilterPairDataset(Dataset):
    """训练集近邻 spot 对 Dataset（表达为原始 log1p_cpm）。

    Parameters
    ----------
    n_neighbors:
        每个 spot 取多少个空间近邻组成配对（**不含自身**）。
    """

    def __init__(
        self,
        h5ad_path: PathLike,
        feature_key: Optional[str] = None,
        gene_list: Union[PathLike, Sequence[str], None] = None,
        n_neighbors: int = 6,
        max_distance: Optional[float] = None,
        bidirectional: bool = True,
        split: str = "train",
    ):
        super().__init__()
        self.h5ad_path = Path(h5ad_path)
        self.n_neighbors = int(n_neighbors)
        self.max_distance = max_distance
        self.bidirectional = bool(bidirectional)
        self.split = split

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
                    f"[GapFilterPairDataset] gene_list 中有 {missing} 个基因不在 adata 中，已忽略"
                )
            if not keep:
                raise ValueError("gene_list 与 adata.var_names 无交集")
            gene_mask = adata.var_names.isin(keep)
            self.gene_names = np.asarray(adata.var_names[gene_mask], dtype=str)

        split_mask = adata.obs["split"].astype(str).values == self.split
        # 不含自身时至少需要 n_neighbors+1 个 spot
        if split_mask.sum() < self.n_neighbors + 1:
            raise ValueError(
                f"split='{self.split}' 的 spot 数 ({split_mask.sum()}) "
                f"不足以构建 n_neighbors={self.n_neighbors}（不含自身）的近邻对"
            )

        adata_sub = adata[split_mask].copy()
        self.coords = np.asarray(adata_sub.obsm["spatial"], dtype=np.float64)
        self.features = np.asarray(adata_sub.obsm[self.feature_key], dtype=np.float32)
        self.expr = _to_dense(adata_sub.layers["log1p_cpm"][:, gene_mask]).astype(
            np.float32
        )
        self.spot_ids = np.asarray(adata_sub.obs_names, dtype=str)
        self.n_spots = len(self.spot_ids)

        if self.features.shape[1] % 2 != 0:
            raise ValueError(
                f"特征维 {self.features.shape[1]} 不是偶数，无法拆成 local|contextual"
            )

        # 默认用全部 spot 构图（兼容直接遍历）；CV 时用 subset_pairs
        self.pairs = self.build_pairs(np.arange(self.n_spots))
        if len(self.pairs) == 0:
            raise RuntimeError("未构建出任何近邻对，请检查 n_neighbors / max_distance")

        print(
            f"[GapFilterPairDataset] spots={self.n_spots}, "
            f"genes={len(self.gene_names)}, pairs={len(self.pairs)}, "
            f"feature_key={self.feature_key}, feature_dim={self.features.shape[1]}"
        )

    def build_pairs(self, spot_indices: np.ndarray) -> List[tuple]:
        """仅在给定 spot 子集内构图（两端都必须属于该子集）。"""
        spot_indices = np.asarray(spot_indices, dtype=np.int64)
        if spot_indices.size < self.n_neighbors + 1:
            return []

        sub_coords = self.coords[spot_indices]
        k = min(self.n_neighbors + 1, len(spot_indices))  # +1 以便丢掉自身
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(sub_coords)
        distances, indices = nn.kneighbors(sub_coords)

        pairs: List[tuple] = []
        seen = set()
        for local_i in range(len(spot_indices)):
            i = int(spot_indices[local_i])
            for rank in range(1, k):  # 跳过自身
                local_j = int(indices[local_i, rank])
                j = int(spot_indices[local_j])
                d = float(distances[local_i, rank])
                if self.max_distance is not None and d > self.max_distance:
                    continue
                edge = (i, j)
                if edge not in seen:
                    pairs.append(edge)
                    seen.add(edge)
                if self.bidirectional:
                    rev = (j, i)
                    if rev not in seen:
                        pairs.append(rev)
                        seen.add(rev)
        return pairs

    def subset_pairs(self, spot_indices: np.ndarray) -> _PairView:
        """按 spot 子集构建 pair 视图（用于 spot 级 CV）。"""
        pairs = self.build_pairs(spot_indices)
        if len(pairs) == 0:
            raise RuntimeError(
                f"spot 子集大小={len(spot_indices)} 未能构建近邻对，"
                f"请减小 n_folds 或 n_neighbors"
            )
        return _PairView(self, pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j = self.pairs[index]
        return {
            "x1": torch.from_numpy(self.features[i]),
            "x2": torch.from_numpy(self.features[j]),
            "y1": torch.from_numpy(self.expr[i]),
            "y2": torch.from_numpy(self.expr[j]),
            "idx1": i,
            "idx2": j,
        }

    def save_gene_names(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, gene_names=self.gene_names)
