"""Xenium 数据预处理器。

流程（参考 ``notebooks/Xenium_prep_s1.ipynb``）：

1. 解压 ``*_outs.zip`` 到 ``data/interim``（运算量大的中间产物）；
2. 读取 H&E 图像与像素尺寸，DAPI 像素尺寸取自 ``experiment.xenium`` 的 ``pixel_size``；
3. 读取转录本，过滤阴性对照与低 QV；
4. 生成微米坐标下的模拟 Visium 网格（train + 两套间隙 test）；
5. 用 KDTree + bincount 把转录本聚合成 spot；
6. 微米坐标 -> DAPI 像素 -> H&E 像素（仿射矩阵逆变换），过滤负坐标；
7. HEG∩HVG 候选池 + Moran's I 选 SVG；
8. 子集到 min_cells 过滤后的基因并保存 ``expr.h5ad``；train/test 叠加 H&E 的 QC 图保存到 interim。

``--from-interim``：跳过 1–6，直接加载 interim 的 ``aggregated_spots.h5ad``。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from preprocessing import archives, utils
from preprocessing.base import BaseProcessor, PreprocessError, SampleSpec

_TRANSCRIPT_COLS = ["feature_name", "x_location", "y_location", "qv"]


def _read_transcripts(outs_root):
    """优先读取 transcripts.parquet（更快），否则回退 transcripts.csv.gz。"""
    parquet = archives.find_one(outs_root, "transcripts.parquet")
    if parquet is not None:
        df = pd.read_parquet(str(parquet), columns=_TRANSCRIPT_COLS)
        if df["feature_name"].dtype == object and isinstance(
            df["feature_name"].iloc[0], bytes
        ):
            df["feature_name"] = df["feature_name"].str.decode("utf-8")
        return df
    csv = archives.find_one(outs_root, "transcripts.csv.gz")
    if csv is not None:
        return pd.read_csv(str(csv), usecols=_TRANSCRIPT_COLS)
    raise PreprocessError(f"未找到 transcripts 文件: {outs_root}")


def _dapi_to_he(dapi_px: np.ndarray, alignment_csv_path) -> np.ndarray:
    """用仿射矩阵的逆变换把 DAPI 像素坐标转换为 H&E 像素坐标。"""
    m = np.loadtxt(str(alignment_csv_path), delimiter=",")
    if m.shape != (3, 3):
        raise PreprocessError(f"仿射矩阵必须是 3x3，实际为 {m.shape}")
    m_inv = np.linalg.inv(m)
    homog = np.hstack([dapi_px, np.ones((dapi_px.shape[0], 1))])
    return (homog @ m_inv.T)[:, :2]


class XeniumProcessor(BaseProcessor):
    data_type = "Xenium"

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
        import anndata as ad

        # 1. 解压 outs.zip 到 interim
        outs_zip = sample.files.get("outs_zip")
        outs_root = sample.interim_dir / "outs_extracted"
        if outs_zip is not None and outs_zip.exists():
            archives.extract_archive(
                outs_zip, outs_root, overwrite=self.config.overwrite
            )
        elif (sample.raw_dir / "outs").exists():
            outs_root = sample.raw_dir
        else:
            raise PreprocessError(f"未找到 outs 压缩包或目录: {sample.name}")

        # 2. 图像 + 像素尺寸
        alignment_path = sample.files.get("alignment")
        if he_path is None or not he_path.exists():
            raise PreprocessError(f"未找到 H&E 图像: {sample.name}")
        if alignment_path is None or not alignment_path.exists():
            raise PreprocessError(f"未找到 H&E 对齐矩阵 csv: {sample.name}")

        experiment_path = archives.find_one(outs_root, "experiment.xenium")
        pixel_size_dapi = None
        if experiment_path is not None:
            with open(experiment_path) as f:
                pixel_size_dapi = float(json.load(f).get("pixel_size"))
        if not pixel_size_dapi:
            raise PreprocessError(
                f"无法从 experiment.xenium 获取 DAPI 像素尺寸: {sample.name}"
            )
        pixel_size_he = utils.read_pixel_size_um(he_path)
        image = utils.read_tiff_image(he_path)
        utils.save_image_and_pixel_size(image, pixel_size_he, sample.processed_dir)

        # 3. 读取并过滤转录本
        df = _read_transcripts(outs_root)
        n_genes_all = df["feature_name"].nunique()
        df = df[~df["feature_name"].str.contains(self.config.xenium_neg_control_pattern)]
        df = df[df["qv"] > self.config.xenium_qv_threshold]
        logger.info(
            f"过滤后转录本={len(df)}, 基因数 {n_genes_all} -> {df['feature_name'].nunique()}"
        )

        # 4. 微米坐标下的模拟 Visium 网格
        xmin, xmax = df["x_location"].min(), df["x_location"].max()
        ymin, ymax = df["y_location"].min(), df["y_location"].max()
        grids = utils.build_train_test_grids(
            xmin, xmax, ymin, ymax, self.config.spot, unit_scale=1.0
        )

        # 5. 聚合转录本 -> spot
        gene_codes, gene_names = pd.factorize(df["feature_name"], sort=True)
        xy = df[["x_location", "y_location"]].to_numpy(dtype=np.float32)
        spot_centers = grids[["x_center", "y_center"]].to_numpy(dtype=np.float32)
        radius_um = self.config.spot.spot_diameter_um / 2.0
        logger.info("聚合 Xenium 转录本到模拟 spot ...")
        X = utils.aggregate_points_to_spots(
            xy, gene_codes, gene_names.size, spot_centers, radius_um
        )

        obs = grids.set_index("spot_id").copy()
        var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obsm["spatial_um"] = obs[["x_center", "y_center"]].to_numpy()

        # 6. 坐标变换：µm -> DAPI px -> H&E px
        dapi_px = adata.obsm["spatial_um"] / pixel_size_dapi
        he_px = _dapi_to_he(dapi_px, alignment_path)
        adata.obsm["spatial"] = he_px
        keep = np.all(adata.obsm["spatial"] >= 0, axis=1)
        adata = adata[keep].copy()
        logger.info(f"坐标变换后保留 spot: {adata.n_obs}")
        return adata, image
