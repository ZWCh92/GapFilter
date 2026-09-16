"""
从 processed 目录中加载提取完特征的 h5ad 文件 (expr_with_HE_features.h5ad)，
构建 Dataset，用于训练模型。

加载的是 layers['log1p_cpm'] 的表达量。

需要将数据进行逐基因 min-max 归一化，并保存每个基因的归一化参数，
便于后续预测时恢复表达尺度。

return:
1. 从训练集上获取位置相近的两个 spot 的 H&E feature，分别记为 x1, x2；
2. 对应 spot 的待预测基因的 min-max 后的表达量（放缩系数也要保存），记为 y1, y2；
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

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
        genes = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        return genes
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
        raise ValueError(
            f"发现多个特征键 {candidates}，请显式指定 feature_key"
        )
    raise KeyError("adata.obsm 中未找到 '*_features'，请指定 feature_key")


class GapFilterPairDataset(Dataset):
    """训练集上近邻 spot 对的 H&E feature + 表达 Dataset。

    Parameters
    ----------
    h5ad_path:
        ``expr_with_HE_features.h5ad`` 路径。
    feature_key:
        ``adata.obsm`` 中 H&E 特征键；默认自动推断 ``*_features``。
    gene_list:
        待预测基因名列表，或基因清单文件路径；默认使用全部基因。
    n_neighbors:
        每个 train spot 取多少个空间近邻组成配对（含自身；例如 6 则含 1 个自身 + 5 个邻居）。
    max_distance:
        近邻最大距离（与 ``obsm['spatial']`` 同单位）；``None`` 表示不限制。
        自身配对距离为 0，不受该阈值影响。
    bidirectional:
        若为 True，每个异向近邻边额外加入反向对 ``(j, i)``（自身对不变）。
    split:
        用于构图与 min-max 统计的划分，默认 ``train``。
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
        eps: float = 1e-8,
    ):
        super().__init__()
        self.h5ad_path = Path(h5ad_path)
        self.n_neighbors = int(n_neighbors)
        self.max_distance = max_distance
        self.bidirectional = bool(bidirectional)
        self.split = split
        self.eps = float(eps)

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
                print(f"[GapFilterPairDataset] gene_list 中有 {missing} 个基因不在 adata 中，已忽略")
            if not keep:
                raise ValueError("gene_list 与 adata.var_names 无交集")
            gene_mask = adata.var_names.isin(keep)
            self.gene_names = np.asarray(adata.var_names[gene_mask], dtype=str)

        split_mask = adata.obs["split"].astype(str).values == self.split
        if split_mask.sum() < self.n_neighbors:
            raise ValueError(
                f"split='{self.split}' 的 spot 数 ({split_mask.sum()}) "
                f"不足以构建 n_neighbors={self.n_neighbors}（含自身）的近邻对"
            )

        # 仅保留目标 split，索引映射到子集
        adata_sub = adata[split_mask].copy()
        coords = np.asarray(adata_sub.obsm["spatial"], dtype=np.float64)
        features = np.asarray(adata_sub.obsm[self.feature_key], dtype=np.float32)
        expr = _to_dense(adata_sub.layers["log1p_cpm"][:, gene_mask]).astype(np.float32)

        # 逐基因 min-max（在当前 split 上拟合，供训练与后续反归一化）
        self.gene_min = expr.min(axis=0)
        self.gene_max = expr.max(axis=0)
        denom = np.maximum(self.gene_max - self.gene_min, self.eps)
        expr_norm = (expr - self.gene_min) / denom

        self.features = features
        self.expr_norm = expr_norm
        self.coords = coords
        self.spot_ids = np.asarray(adata_sub.obs_names, dtype=str)

        self.pairs = self._build_neighbor_pairs(coords)
        if len(self.pairs) == 0:
            raise RuntimeError("未构建出任何近邻对，请检查 n_neighbors / max_distance")

        print(
            f"[GapFilterPairDataset] spots={len(self.spot_ids)}, "
            f"genes={len(self.gene_names)}, pairs={len(self.pairs)}, "
            f"feature_key={self.feature_key}, feature_dim={features.shape[1]}"
        )

    def _build_neighbor_pairs(self, coords: np.ndarray) -> List[tuple]:
        n = coords.shape[0]
        k = min(self.n_neighbors, n)  # 含自身
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

        pairs: List[tuple] = []
        seen = set()
        for i in range(n):
            for rank in range(k):  # rank=0 为自身
                j = int(indices[i, rank])
                d = float(distances[i, rank])
                if i != j and self.max_distance is not None and d > self.max_distance:
                    continue
                edge = (i, j)
                if edge not in seen:
                    pairs.append(edge)
                    seen.add(edge)
                if self.bidirectional and i != j:
                    rev = (j, i)
                    if rev not in seen:
                        pairs.append(rev)
                        seen.add(rev)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        i, j = self.pairs[index]
        return {
            "x1": torch.from_numpy(self.features[i]),
            "x2": torch.from_numpy(self.features[j]),
            "y1": torch.from_numpy(self.expr_norm[i]),
            "y2": torch.from_numpy(self.expr_norm[j]),
            "idx1": i,
            "idx2": j,
        }

    def get_scale_params(self) -> dict:
        """返回逐基因 min-max 参数，便于预测阶段恢复表达尺度。"""
        return {
            "gene_names": self.gene_names.copy(),
            "gene_min": self.gene_min.copy(),
            "gene_max": self.gene_max.copy(),
            "eps": self.eps,
        }

    def denormalize(self, y_norm: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """将 min-max 后的表达还原到 log1p_cpm 尺度。"""
        if isinstance(y_norm, torch.Tensor):
            y_norm = y_norm.detach().cpu().numpy()
        y_norm = np.asarray(y_norm, dtype=np.float32)
        denom = np.maximum(self.gene_max - self.gene_min, self.eps)
        return y_norm * denom + self.gene_min

    def save_scale_params(self, path: PathLike) -> None:
        """保存归一化参数到 .npz。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            gene_names=self.gene_names,
            gene_min=self.gene_min,
            gene_max=self.gene_max,
            eps=np.asarray(self.eps),
        )
