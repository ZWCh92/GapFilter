"""VisiumHD 数据预处理器。

流程（参考 ``notebooks/VisiumHD_prep_s1.ipynb``）：

1. 解压 ``*_binned_outputs.tar.gz`` 到 ``data/interim``（运算量大的中间产物）；
2. 读取 H&E 图像与像素尺寸（优先 ``scalefactors_json.json`` 的 microns_per_pixel）；
3. 读取 8µm bin 的表达矩阵与坐标，对齐、过滤；
4. 生成模拟 Visium 网格（train + 两套间隙 test，像素坐标）；
5. 用 KDTree + 稀疏矩阵乘法把 HD bins 聚合成伪 Visium spot；
6. HEG∩HVG 候选池 + Moran's I 选 SVG；
7. 子集到 min_cells 过滤后的基因并保存 ``expr.h5ad``；train/test 叠加 H&E 的 QC 图保存到 interim。

``--from-interim``：跳过 1–5，直接加载 interim 的 ``pseudo_visium_aggregated.h5ad``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from preprocessing import archives, utils
from preprocessing.base import BaseProcessor, PreprocessError, SampleSpec


class VisiumHDProcessor(BaseProcessor):
    data_type = "VisiumHD"

    def process_sample(self, sample: SampleSpec) -> None:
        import anndata as ad
        import scanpy as sc

        gene_cfg = self.config.gene_selection_for(self.data_type)
        he_path = sample.files.get("he")

        if self.config.from_interim:
            adata = utils.load_interim_adata(sample)
            image = utils.load_he_image_for_sample(sample, he_path)
        else:
            adata, image = self._build_from_raw(sample, he_path)
            utils.save_interim_adata(adata, sample)

        # 基因选择（SVG 清单 -> processed，moranI.csv -> interim）
        result = utils.select_genes(
            adata, gene_cfg, sample.processed_dir, sample.interim_dir
        )

        # QC：train/test spot 叠加 H&E，保存到 interim
        utils.save_split_qc_overlays(
            adata,
            image,
            sample.interim_dir,
            result["top_gene"],
            coords_key="spatial",
            max_image_dim=self.config.qc_max_image_dim,
        )

        # 保存最终表达数据
        utils.finalize_and_save_adata(
            adata,
            result["keep_genes"],
            sample.output_h5ad,
            add_log1p_cpm_layer=self.config.save_log1p_cpm_layer,
        )

    def _build_from_raw(self, sample: SampleSpec, he_path):
        import anndata as ad
        import scanpy as sc

        bin_dir_name = self.config.visiumhd_bin

        # 1. 解压 binned_outputs 到 interim
        binned_tar = sample.files.get("binned_tar")
        binned_root = sample.interim_dir / "binned_outputs_extracted"
        if binned_tar is not None and binned_tar.exists():
            archives.extract_archive(
                binned_tar, binned_root, overwrite=self.config.overwrite
            )
        elif (sample.raw_dir / "binned_outputs").exists():
            binned_root = sample.raw_dir
        else:
            raise PreprocessError(f"未找到 binned_outputs 压缩包或目录: {sample.name}")

        counts_path = archives.find_one(
            binned_root, f"{bin_dir_name}/filtered_feature_bc_matrix.h5"
        )
        positions_path = archives.find_one(
            binned_root, f"{bin_dir_name}/spatial/tissue_positions.parquet"
        )
        scalefactors_path = archives.find_one(
            binned_root, f"{bin_dir_name}/spatial/scalefactors_json.json"
        )
        if counts_path is None or positions_path is None:
            raise PreprocessError(
                f"{bin_dir_name} 下缺少表达矩阵或坐标文件: {sample.name}"
            )

        # 2. 图像 + 像素尺寸
        if he_path is None or not he_path.exists():
            raise PreprocessError(f"未找到 H&E 图像: {sample.name}")
        pixel_size = None
        if scalefactors_path is not None:
            pixel_size = utils.read_microns_per_pixel_from_scalefactors(scalefactors_path)
        if pixel_size is None:
            pixel_size = utils.read_pixel_size_um(he_path)
        if pixel_size is None:
            raise PreprocessError(
                f"无法确定像素尺寸(µm/px)，VisiumHD 网格聚合依赖该值: {sample.name}"
            )
        image = utils.read_tiff_image(he_path)
        utils.save_image_and_pixel_size(image, pixel_size, sample.processed_dir)

        # 3. 读取并对齐
        counts = sc.read_10x_h5(str(counts_path))
        coords_df = pd.read_parquet(str(positions_path))
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
        counts = counts[common].copy()
        coords_df = coords_df.loc[common]
        if not counts.var_names.is_unique:
            counts.var_names_make_unique()

        # 4. 生成像素坐标下的模拟 Visium 网格
        unit_scale = 1.0 / pixel_size  # µm -> px
        xs = coords_df["pxl_col_in_fullres"].to_numpy()
        ys = coords_df["pxl_row_in_fullres"].to_numpy()
        grids = utils.build_train_test_grids(
            xs.min(), xs.max(), ys.min(), ys.max(), self.config.spot, unit_scale
        )

        # 5. 聚合 HD bins -> 伪 Visium spot
        bin_xy = coords_df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(
            dtype=np.float32
        )
        radius_px = self.config.spot.spot_diameter_um / 2.0 * unit_scale
        spot_centers = grids[["x_center", "y_center"]].to_numpy(dtype=np.float32)
        logger.info("聚合 VisiumHD bins 到伪 Visium spot ...")
        new_X, bin_count = utils.aggregate_bins_to_spots(
            bin_xy, counts.X, spot_centers, radius_px
        )

        obs = grids.set_index("spot_id").copy()
        obs["bin_count"] = bin_count
        adata = ad.AnnData(X=new_X, obs=obs, var=counts.var.copy())
        adata.obsm["spatial"] = obs[["x_center", "y_center"]].to_numpy()
        adata = adata[adata.obs["bin_count"] > 0].copy()
        logger.info(f"保留有效伪 spot: {adata.n_obs} (覆盖至少 1 个 bin)")
        return adata, image
