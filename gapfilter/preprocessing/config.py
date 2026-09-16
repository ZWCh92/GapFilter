"""预处理超参数配置。

所有可调参数集中在此，方便按数据类型覆盖默认值。默认值与 ``notebooks/`` 下的三个
预处理示例保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class GeneSelectionConfig:
    """基因选择（HEG ∩ HVG 构建候选池 + Moran's I 选 SVG）参数。

    流程：
    1. 分别取 top-``candidate_pool_size`` 的高表达基因 (HEG) 与高变异基因 (HVG)，取交集；
    2. 从交集中按离散度取 top-``pool_topk`` 作为候选池；
    3. 在候选池上计算 Moran's I，按 I 从高到低输出 ``svg_targets`` 指定数目的 SVG 清单
       （若候选池基因数 num 小于某目标，则该清单退化为 ``SVG_top_{num}.txt``）。
    """

    candidate_pool_size: int = 8000  # HEG / HVG 各自的 top-N，用于取交集
    pool_topk: int = 5000  # 从 HEG∩HVG 交集中取 top-K 作为候选池
    svg_targets: Tuple[int, ...] = (200, 2000)  # 输出的 SVG 基因数目标
    min_cells: int = 20
    n_neighs: int = 6
    n_perms: int = 50
    n_jobs: int = -1


@dataclass(frozen=True)
class SpotGeometryConfig:
    """模拟 Visium spot 的几何参数（单位：微米）。"""

    spot_diameter_um: float = 55.0
    pitch_um: float = 100.0
    mode: str = "hex"
    keep: str = "full"


@dataclass(frozen=True)
class PreprocessConfig:
    """单次预处理运行的完整配置。"""

    # Xenium 转录本质控
    xenium_qv_threshold: float = 20.0
    xenium_neg_control_pattern: str = "NegControlCodeword"

    # 几何与基因选择
    spot: SpotGeometryConfig = field(default_factory=SpotGeometryConfig)
    gene_selection: GeneSelectionConfig = field(default_factory=GeneSelectionConfig)

    # VisiumHD 使用的分辨率层级（bin 大小）
    visiumhd_bin: str = "square_008um"

    # 行为开关
    overwrite: bool = False  # 若最终产物已存在是否重跑
    from_interim: bool = False  # True=从 interim 聚合缓存续跑；False=从 raw 全量重跑
    save_log1p_cpm_layer: bool = True  # 是否额外保存 log1p-CPM 层

    # QC 叠加图的最大边长（像素），用于对大图下采样后再绘制
    qc_max_image_dim: int = 2000

    # 每种数据类型对基因选择参数的覆盖（数据类型 -> 覆盖字段）。
    # 基因数目标已内置 min() 裁剪，Xenium 等小基因组无需特殊覆盖。
    per_type_gene_selection: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def gene_selection_for(self, data_type: str) -> GeneSelectionConfig:
        """返回某数据类型实际生效的基因选择配置。"""
        overrides = self.per_type_gene_selection.get(data_type, {})
        if not overrides:
            return self.gene_selection
        return replace(self.gene_selection, **overrides)
