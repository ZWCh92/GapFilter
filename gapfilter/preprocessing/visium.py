"""Visium 数据预处理器。

流程（参考 ``notebooks/Visium_prep_s1.ipynb``）：

1. 读取 H&E 图像与像素尺寸，保存到 processed；
2. 读取基因表达矩阵 (``filtered_feature_bc_matrix.h5``) 与空间坐标
   (``tissue_positions*.csv/parquet``)，对齐 barcode、过滤负坐标、基因名去重；
3. 用 ``array_row + array_col`` 棋盘格划分 train/test；
4. HEG∩HVG 候选池 + Moran's I 选 SVG；
5. 子集到 min_cells 过滤后的基因并保存 ``expr.h5ad``；train/test 叠加 H&E 的 QC 图保存到 interim。

``--from-interim``：跳过 1–3，直接加载 interim 的 ``aligned_spots.h5ad``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from preprocessing import utils
from preprocessing.base import BaseProcessor, PreprocessError, SampleSpec

_POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


def _read_positions(path) -> pd.DataFrame:
    """读取 tissue positions（自动兼容有/无表头的 csv 及 parquet）。"""
    path = str(path)
    if path.endswith(".parquet"):
        coords = pd.read_parquet(path)
    else:
        head = pd.read_csv(path, nrows=1)
        if "barcode" in head.columns:
            coords = pd.read_csv(path)
        else:
            coords = pd.read_csv(path, header=None)
            coords.columns = _POSITION_COLUMNS
    return coords


class VisiumProcessor(BaseProcessor):
    data_type = "Visium"

    def process_sample(self, sample: SampleSpec) -> None:
        gene_cfg = self.config.gene_selection_for(self.data_type)
        he_path = sample.files.get("he")

        if self.config.from_interim:
            adata = utils.load_interim_adata(sample)
            image = utils.load_he_image_for_sample(sample, he_path)
        else:
            adata, image = self._build_from_raw(sample, he_path)
            utils.save_interim_adata(adata, sample)

        result = utils.select_genes(
            adata, gene_cfg, sample.processed_dir, sample.interim_dir
        )

        utils.save_split_qc_overlays(
            adata,
            image,
            sample.interim_dir,
            result["top_gene"],
            coords_key="spatial",
            max_image_dim=self.config.qc_max_image_dim,
        )

        utils.finalize_and_save_adata(
            adata,
            result["keep_genes"],
            sample.output_h5ad,
            add_log1p_cpm_layer=self.config.save_log1p_cpm_layer,
        )

    def _build_from_raw(self, sample: SampleSpec, he_path):
        import scanpy as sc

        # 1. 图像 + 像素尺寸
        if he_path is None or not he_path.exists():
            raise PreprocessError(f"未找到 H&E 图像: {sample.name}")
        image = utils.read_tiff_image(he_path)
        pixel_size = self._resolve_pixel_size(sample, he_path)
        utils.save_image_and_pixel_size(image, pixel_size, sample.processed_dir)

        # 2. 表达矩阵 + 坐标
        counts_path = sample.files.get("counts")
        positions_path = sample.files.get("positions")
        if counts_path is None or not counts_path.exists():
            raise PreprocessError(f"未找到基因表达矩阵 (.h5): {sample.name}")
        if positions_path is None or not positions_path.exists():
            raise PreprocessError(
                f"未找到空间坐标文件 (tissue_positions)。DLPFC 等数据集下载时可能未包含"
                f"坐标文件，请补齐后再处理: {sample.name}"
            )

        counts = sc.read_10x_h5(str(counts_path))
        coords_df = _read_positions(positions_path)

        valid = (coords_df["pxl_col_in_fullres"] >= 0) & (
            coords_df["pxl_row_in_fullres"] >= 0
        )
        coords_df = coords_df[valid]
        if "barcode" in coords_df.columns:
            coords_df = coords_df.set_index("barcode")

        common = counts.obs_names.intersection(coords_df.index)
        logger.info(
            f"counts={counts.shape}, coords={coords_df.shape}, 交集={len(common)}"
        )
        if len(common) == 0:
            raise PreprocessError(f"表达矩阵与坐标 barcode 无交集: {sample.name}")
        counts = counts[common].copy()
        coords_df = coords_df.loc[common]
        if not counts.var_names.is_unique:
            logger.warning("检测到重复基因名，执行 var_names_make_unique()。")
            counts.var_names_make_unique()

        # 3. 棋盘格划分
        adata = counts
        if not {"array_row", "array_col"}.issubset(coords_df.columns):
            raise PreprocessError(
                f"坐标文件缺少 array_row/array_col，无法棋盘格划分: {sample.name}"
            )
        adata.obs["array_row"] = coords_df["array_row"].values
        adata.obs["array_col"] = coords_df["array_col"].values
        adata.obsm["spatial"] = coords_df[
            ["pxl_col_in_fullres", "pxl_row_in_fullres"]
        ].values

        s = adata.obs["array_row"] + adata.obs["array_col"]
        adata.obs["split"] = "train"
        if np.all((s % 2) == 0):
            mask_gt = ((s // 2) % 2) != 0
        else:
            mask_gt = (s % 2) != 0
        adata.obs.loc[mask_gt.values, "split"] = "test"
        logger.info(f"划分统计:\n{adata.obs['split'].value_counts()}")
        return adata, image

    def _resolve_pixel_size(self, sample: SampleSpec, he_path):
        sf = sample.files.get("scalefactors")
        if sf is not None and sf.exists():
            ps = utils.read_microns_per_pixel_from_scalefactors(sf)
            if ps:
                return ps
        ps = utils.read_pixel_size_um(he_path)
        if ps is None:
            logger.warning(f"无法自动解析像素尺寸，将记为 unknown: {he_path.name}")
        return ps
