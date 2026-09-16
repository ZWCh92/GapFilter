"""预处理共享工具函数：图像/像素尺寸读取、Visium 网格生成、基因选择、结果保存。

这些函数从 ``notebooks/`` 下的三个预处理 notebook 中抽取、整理而来，去除了交互式
可视化，并做了健壮性增强（参数越界裁剪、像素尺寸自动解析等）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger

from preprocessing.config import GeneSelectionConfig, SpotGeometryConfig
from preprocessing.base import PreprocessError, SampleSpec

# 各数据类型在 interim 中缓存的「聚合/对齐后 AnnData」文件名
INTERIM_ADATA_NAMES = {
    "Visium": "aligned_spots.h5ad",
    "VisiumHD": "pseudo_visium_aggregated.h5ad",
    "Xenium": "aggregated_spots.h5ad",
}


def interim_adata_path(sample: SampleSpec) -> Path:
    """返回该样本 interim 聚合/对齐 AnnData 路径。"""
    name = INTERIM_ADATA_NAMES.get(sample.data_type)
    if name is None:
        raise PreprocessError(f"未知数据类型，无法定位 interim 缓存: {sample.data_type}")
    return sample.interim_dir / name


def load_interim_adata(sample: SampleSpec):
    """从 interim 加载聚合/对齐 AnnData；不存在则报错。"""
    import scanpy as sc

    path = interim_adata_path(sample)
    if not path.exists():
        raise PreprocessError(
            f"--from-interim 但未找到缓存: {path}。"
            "请先不加该参数完整跑一遍，或检查路径。"
        )
    logger.info(f"从 interim 续跑，加载: {path}")
    return sc.read_h5ad(str(path))


def save_interim_adata(adata, sample: SampleSpec) -> Path:
    """将聚合/对齐后的 AnnData 写入 interim，供后续 --from-interim 续跑。"""
    path = interim_adata_path(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(path))
    logger.info(f"聚合/对齐中间结果已缓存: {path}  (shape={adata.shape})")
    return path


def load_he_image_for_sample(sample: SampleSpec, he_raw_path: Optional[Path] = None):
    """加载 H&E：优先 processed/HE.tif，否则 raw 路径。"""
    processed_he = sample.processed_dir / "HE.tif"
    if processed_he.exists():
        logger.info(f"使用已有 processed H&E: {processed_he}")
        return read_tiff_image(processed_he)
    if he_raw_path is not None and Path(he_raw_path).exists():
        return read_tiff_image(he_raw_path)
    raise PreprocessError(
        f"未找到 H&E 图像（processed 与 raw 均无）: {sample.name}"
    )

# ---------------------------------------------------------------------------
# 图像与像素尺寸
# ---------------------------------------------------------------------------


def read_tiff_image(path: Path) -> np.ndarray:
    """读取 TIFF/OME-TIFF/BTF 的第一页为 numpy 数组。"""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        return tif.pages[0].asarray()


def _pixel_size_from_ome_xml(ome_xml: str) -> Optional[float]:
    """从 OME-XML 元数据解析 PhysicalSizeX（统一换算为微米）。"""
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return None
    for pixels in root.iter():
        if not pixels.tag.endswith("Pixels"):
            continue
        size = pixels.attrib.get("PhysicalSizeX")
        if size is None:
            continue
        try:
            value = float(size)
        except ValueError:
            continue
        unit = pixels.attrib.get("PhysicalSizeXUnit", "µm")
        factor = {
            "µm": 1.0,
            "um": 1.0,
            "micron": 1.0,
            "microns": 1.0,
            "nm": 1e-3,
            "mm": 1e3,
            "cm": 1e4,
            "m": 1e6,
        }.get(unit, 1.0)
        return value * factor
    return None


def _pixel_size_from_tags(tif) -> Optional[float]:
    """从 TIFF 的 XResolution / ResolutionUnit 标签推算像素微米数。"""
    page = tif.pages[0]
    tags = page.tags
    if "XResolution" not in tags or "ResolutionUnit" not in tags:
        return None
    xres = tags["XResolution"].value
    if isinstance(xres, tuple):  # rational: (numerator, denominator)
        if xres[1] == 0:
            return None
        xres = xres[0] / xres[1]
    if not xres:
        return None
    unit = tags["ResolutionUnit"].value
    unit = getattr(unit, "value", unit)  # enum -> int
    # ResolutionUnit: 2 = inch, 3 = centimeter
    unit_um = {2: 25400.0, 3: 10000.0}.get(int(unit))
    if unit_um is None:
        return None
    return unit_um / xres


def read_pixel_size_um(path: Path) -> Optional[float]:
    """尽力从 TIFF 元数据（OME-XML 优先，其次分辨率标签）解析像素微米数。

    解析失败时返回 ``None``（调用方需自行决定回退策略）。
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        if tif.ome_metadata:
            ps = _pixel_size_from_ome_xml(tif.ome_metadata)
            if ps:
                return ps
        return _pixel_size_from_tags(tif)


def read_microns_per_pixel_from_scalefactors(scalefactors_path: Path) -> Optional[float]:
    """从 spaceranger 的 ``scalefactors_json.json`` 读取全分辨率像素微米数。"""
    if not scalefactors_path.exists():
        return None
    with open(scalefactors_path) as f:
        data = json.load(f)
    if "microns_per_pixel" in data:
        return float(data["microns_per_pixel"])
    # 退路：由 55µm spot 直径与全分辨率 spot 直径像素数推算
    spot_diam_px = data.get("spot_diameter_fullres")
    if spot_diam_px:
        return 55.0 / float(spot_diam_px)
    return None


def save_image_and_pixel_size(
    image: np.ndarray, pixel_size_um: Optional[float], out_dir: Path
) -> None:
    """保存 H&E 图像为 ``HE.tif``，像素尺寸(MPP, µm/px)写入 ``MPP.txt``。"""
    import tifffile

    out_dir.mkdir(parents=True, exist_ok=True)
    he_path = out_dir / "HE.tif"
    tifffile.imwrite(str(he_path), image)
    logger.info(f"H&E 图像已保存: {he_path}")

    mpp_path = out_dir / "MPP.txt"
    mpp_path.write_text("unknown" if pixel_size_um is None else str(pixel_size_um))
    logger.info(f"像素尺寸 MPP(µm/px) 已保存: {mpp_path} -> {pixel_size_um}")


# ---------------------------------------------------------------------------
# 模拟 Visium 网格
# ---------------------------------------------------------------------------


def build_visium_grid(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    spot_diam: float,
    pitch: float,
    mode: str = "hex",
    keep: str = "full",
    eps: float = 1e-9,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    tag: str = "",
) -> pd.DataFrame:
    """带偏移的 Visium 网格生成器，用于生成 Train/Test 互补网格。

    坐标单位由调用方决定（Xenium 用微米，VisiumHD 用像素），只要 ``spot_diam`` /
    ``pitch`` / 边界与之一致即可。返回含 ``spot_id, x_center, y_center`` 的 DataFrame。
    """
    radius = spot_diam / 2.0
    base_start_x = np.floor(min_x / pitch) * pitch
    base_start_y = np.floor(min_y / pitch) * pitch
    start_x = base_start_x + offset_x
    start_y = base_start_y + offset_y

    xs: np.ndarray
    ys: np.ndarray
    if mode == "hex":
        row_step = pitch * np.sqrt(3) / 2.0
        ny = int(np.ceil((max_y - start_y) / row_step)) + 3
        xs_list, ys_list = [], []
        for iy in range(-2, ny):
            y = start_y + iy * row_step
            x_shift = 0.0 if (iy % 2 == 0) else (pitch / 2.0)
            nx = int(np.ceil((max_x - (start_x + x_shift)) / pitch)) + 3
            x_line = (start_x + x_shift) + np.arange(-2, nx) * pitch
            xs_list.append(x_line)
            ys_list.append(np.full_like(x_line, y))
        xs = np.concatenate(xs_list) if xs_list else np.array([])
        ys = np.concatenate(ys_list) if ys_list else np.array([])
    elif mode == "square":
        nx = int(np.ceil((max_x - start_x) / pitch)) + 2
        ny = int(np.ceil((max_y - start_y) / pitch)) + 2
        X = start_x + np.arange(-1, nx) * pitch
        Y = start_y + np.arange(-1, ny) * pitch
        XX, YY = np.meshgrid(X, Y)
        xs, ys = XX.ravel(), YY.ravel()
    else:
        raise ValueError(f"未知的网格模式: {mode}")

    if keep == "center":
        mask = (
            (xs >= min_x - eps)
            & (xs <= max_x + eps)
            & (ys >= min_y - eps)
            & (ys <= max_y + eps)
        )
    elif keep == "full":
        mask = (
            (xs - radius >= min_x - eps)
            & (xs + radius <= max_x + eps)
            & (ys - radius >= min_y - eps)
            & (ys + radius <= max_y + eps)
        )
    else:
        raise ValueError(f"未知的 keep 策略: {keep}")

    xs, ys = xs[mask], ys[mask]
    ids = [f"spot_{tag}_{i}" if tag else f"spot_{i}" for i in range(len(xs))]
    return pd.DataFrame({"spot_id": ids, "x_center": xs, "y_center": ys})


def build_train_test_grids(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    geom: SpotGeometryConfig,
    unit_scale: float = 1.0,
) -> pd.DataFrame:
    """生成训练网格 + 两套间隙测试网格并合并，附带 ``split`` 列。

    参数
    ----
    unit_scale : float
        微米 -> 目标坐标单位的换算系数。Xenium 直接用微米（=1.0）；VisiumHD 用像素
        坐标，则传入 ``1/pixel_size_um``。
    """
    spot_diam = geom.spot_diameter_um * unit_scale
    pitch = geom.pitch_um * unit_scale
    dx = pitch / 2.0
    dy = pitch * np.sqrt(3) / 6.0

    common = dict(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        spot_diam=spot_diam,
        pitch=pitch,
        mode=geom.mode,
        keep=geom.keep,
    )
    df_train = build_visium_grid(**common, offset_x=0.0, offset_y=0.0, tag="train")
    df_train["split"] = "train"
    df_gap1 = build_visium_grid(**common, offset_x=dx, offset_y=dy, tag="gap1")
    df_gap1["split"] = "test"
    df_gap2 = build_visium_grid(**common, offset_x=dx, offset_y=-dy, tag="gap2")
    df_gap2["split"] = "test"

    grids = pd.concat([df_train, df_gap1, df_gap2], ignore_index=True)
    logger.info(
        f"网格生成: train={len(df_train)}, gap1={len(df_gap1)}, "
        f"gap2={len(df_gap2)}, total={len(grids)}"
    )
    return grids


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def aggregate_points_to_spots(
    point_xy: np.ndarray,
    gene_codes: np.ndarray,
    n_genes: int,
    spot_centers: np.ndarray,
    radius: float,
) -> sp.csr_matrix:
    """将离散分子点（Xenium 转录本）按半径聚合到 spot，返回 (n_spots, n_genes) 计数。"""
    from scipy.spatial import cKDTree

    tree = cKDTree(point_xy)
    hits = tree.query_ball_point(spot_centers, r=float(radius))

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    for r, hit_idx in enumerate(hits):
        if not hit_idx:
            continue
        gcodes = gene_codes[np.fromiter(hit_idx, dtype=np.int64, count=len(hit_idx))]
        cnt = np.bincount(gcodes, minlength=n_genes)
        nz = np.nonzero(cnt)[0]
        if nz.size == 0:
            continue
        rows.extend([r] * nz.size)
        cols.extend(nz.tolist())
        data.extend(cnt[nz].astype(np.int32).tolist())

    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.int32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(len(spot_centers), n_genes),
    )


def aggregate_bins_to_spots(
    bin_xy: np.ndarray,
    bin_X: sp.spmatrix,
    spot_centers: np.ndarray,
    radius: float,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """将 VisiumHD bins 按半径聚合到 spot（稀疏连接矩阵乘法）。

    返回聚合后的 (n_spots, n_genes) 计数矩阵，以及每个 spot 覆盖的 bin 数。
    """
    from scipy.spatial import cKDTree

    n_bins = bin_xy.shape[0]
    n_spots = spot_centers.shape[0]

    tree = cKDTree(bin_xy)
    hits = tree.query_ball_point(spot_centers, r=float(radius))

    rows: List[int] = []
    cols: List[int] = []
    for i, bin_indices in enumerate(hits):
        if not bin_indices:
            continue
        rows.extend([i] * len(bin_indices))
        cols.extend(bin_indices)
    data = np.ones(len(rows), dtype=np.float32)
    connectivity = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_spots, n_bins)
    )

    if not sp.issparse(bin_X):
        bin_X = sp.csr_matrix(bin_X)
    new_X = connectivity @ bin_X
    bin_count = np.diff(connectivity.indptr)
    return sp.csr_matrix(new_X), bin_count


# ---------------------------------------------------------------------------
# 基因选择（HEG ∩ HVG 候选池 + Moran's I SVG）
# ---------------------------------------------------------------------------


def select_genes(
    adata,
    cfg: GeneSelectionConfig,
    gene_list_dir: Path,
    moran_dir: Path,
) -> dict:
    """选出高表达∩高变异候选池，并在候选池内用 Moran's I 选空间可变基因(SVG)。

    要求 ``adata.obsm['spatial']`` 存在。

    - ``SVG_top_{k}.txt`` 清单（基因按 Moran's I 从高到低排序）写入 ``gene_list_dir``；
    - ``moranI.csv``（完整 Moran's I 结果）写入 ``moran_dir``。

    返回：
    - ``keep_genes``：``min_cells`` 过滤后的基因，用于最终 ``expr.h5ad`` 子集；
    - ``pool``：HEG∩HVG 候选池（仅用于 SVG / Moran's I）；
    - ``svg_ordered``：候选池内按 Moran's I 排序的基因；
    - ``top_gene``：SVG 首位。
    """
    import scanpy as sc
    import squidpy as sq

    gene_list_dir.mkdir(parents=True, exist_ok=True)
    moran_dir.mkdir(parents=True, exist_ok=True)
    adata_work = adata.copy()

    # Step 0: 基础质控 + 归一化（仅用于指标计算，不影响最终保存的原始计数）
    min_cells = min(cfg.min_cells, max(1, adata_work.n_obs // 20))
    sc.pp.filter_genes(adata_work, min_cells=min_cells)
    keep_genes = adata_work.var_names.tolist()
    logger.info(
        f"min_cells={min_cells} 过滤后保留基因数: {len(keep_genes)} "
        f"(原始 {adata.n_vars})"
    )
    sc.pp.normalize_total(adata_work)
    sc.pp.log1p(adata_work)

    n_vars = adata_work.n_vars
    if n_vars == 0:
        raise ValueError("质控后没有剩余基因，请检查数据或降低 min_cells。")

    candidate = min(cfg.candidate_pool_size, n_vars)
    pool_topk = min(cfg.pool_topk, n_vars)

    # Step 1: 计算 means 与 dispersions_norm（对所有基因）
    sc.pp.highly_variable_genes(
        adata_work, flavor="seurat", n_top_genes=min(candidate, n_vars), subset=False
    )
    metrics_df = adata_work.var.copy()

    # Step 2: HEG(高表达) / HVG(高变异) 各取 top-candidate 后取交集
    heg_list = (
        metrics_df.sort_values(by="means", ascending=False).head(candidate).index.tolist()
    )
    hvg_list = (
        metrics_df.sort_values(by="dispersions_norm", ascending=False)
        .head(candidate)
        .index.tolist()
    )
    intersection = list(set(heg_list) & set(hvg_list))
    logger.info(f"HEG ∩ HVG (各 top-{candidate}) 交集基因数: {len(intersection)}")

    # Step 3: 从交集中按离散度取 top-pool_topk 作为候选池（仅用于 SVG）
    pool_genes = (
        metrics_df.loc[intersection]
        .sort_values(by="dispersions_norm", ascending=False)
        .head(pool_topk)
        .index.tolist()
    )
    if not pool_genes:
        raise ValueError("候选池为空，无法继续基因选择。")
    logger.info(f"候选池基因数 (HEG∩HVG top-{pool_topk}): {len(pool_genes)}")

    # Step 4: 在候选池上计算 Moran's I，按 I 从高到低排序
    logger.info(f"在候选池 ({len(pool_genes)} 基因) 上计算 Moran's I ...")
    adata_pool = adata_work[:, pool_genes].copy()
    if "spatial" not in adata_pool.obsm:
        raise KeyError("adata.obsm['spatial'] 缺失，无法计算空间自相关。")
    n_neighs = min(cfg.n_neighs, max(1, adata_pool.n_obs - 1))
    sq.gr.spatial_neighbors(adata_pool, coord_type="generic", n_neighs=n_neighs)
    sq.gr.spatial_autocorr(
        adata_pool, mode="moran", n_perms=cfg.n_perms, n_jobs=cfg.n_jobs
    )
    moran_res = adata_pool.uns["moranI"].sort_values(by="I", ascending=False)
    svg_ordered = moran_res.index.tolist()  # 按 Moran's I 从高到低
    num = len(svg_ordered)

    # Step 5: 输出 SVG 清单（基因数不足目标时退化为 SVG_top_{num}.txt）
    written = set()
    for target in cfg.svg_targets:
        k = min(target, num)
        fname = f"SVG_top_{k}.txt"
        if fname in written:
            continue
        _write_gene_list(gene_list_dir / fname, svg_ordered[:k])
        written.add(fname)
        logger.info(f"已输出 {fname} ({k} 个基因)")

    # moranI.csv 输出到 interim
    moran_res.to_csv(moran_dir / "moranI.csv")
    logger.info(f"moranI.csv 已保存至: {moran_dir}")

    return {
        "keep_genes": keep_genes,
        "pool": pool_genes,
        "svg_ordered": svg_ordered,
        "top_gene": svg_ordered[0] if svg_ordered else None,
    }


def _write_gene_list(path: Path, genes: List[str]) -> None:
    with open(path, "w") as f:
        for g in genes:
            f.write(f"{g}\n")


# ---------------------------------------------------------------------------
# 保存最终 AnnData
# ---------------------------------------------------------------------------


def finalize_and_save_adata(
    adata,
    keep_genes: List[str],
    out_path: Path,
    add_log1p_cpm_layer: bool = True,
) -> None:
    """将 adata 子集到 ``keep_genes``（通常为 min_cells 过滤后的基因），写入 h5ad。

    ``adata.X`` 保留原始计数（与 notebook 一致），归一化结果放入 ``layers['log1p_cpm']``。
    """
    import scanpy as sc

    adata = adata[:, keep_genes].copy()
    if add_log1p_cpm_layer:
        norm = adata.copy()
        sc.pp.normalize_total(norm)
        sc.pp.log1p(norm)
        adata.layers["log1p_cpm"] = norm.X
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(str(out_path))
    logger.success(f"预处理结果已保存: {out_path}  (shape={adata.shape})")


# ---------------------------------------------------------------------------
# QC：train/test spot 与 H&E 叠加显示
# ---------------------------------------------------------------------------


def save_split_qc_overlays(
    adata,
    image: np.ndarray,
    out_dir: Path,
    gene_name: Optional[str],
    coords_key: str = "spatial",
    max_image_dim: int = 2000,
    point_size: float = 4.0,
) -> None:
    """将 train / test spot 分别叠加到 H&E 上并保存 PNG（用于质量检查）。

    为避免全分辨率大图占用过多内存，绘图前对图像做整数步长下采样，坐标同比缩放。
    绘图失败（如缺少 matplotlib）时仅记录警告，不影响主流程。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logger.warning(f"跳过 QC 叠加图（无法导入 matplotlib）: {e}")
        return

    if gene_name is None or gene_name not in list(adata.var_names):
        logger.warning(f"跳过 QC 叠加图（基因不可用）: {gene_name}")
        return
    if coords_key not in adata.obsm:
        logger.warning(f"跳过 QC 叠加图（缺少坐标 {coords_key}）: ")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = image.shape[:2]
    factor = max(1, int(np.ceil(max(H, W) / float(max_image_dim))))
    img_small = image[::factor, ::factor]

    gidx = list(adata.var_names).index(gene_name)
    for split in ("train", "test"):
        sub = adata[adata.obs["split"] == split]
        if sub.n_obs == 0:
            logger.warning(f"split={split} 无 spot，跳过其 QC 图。")
            continue
        coords = np.asarray(sub.obsm[coords_key], dtype=float) / factor
        expr = sub.X[:, gidx]
        expr = expr.toarray().ravel() if sp.issparse(expr) else np.asarray(expr).ravel()
        expr = np.log1p(np.clip(expr, 0, None))
        order = np.argsort(expr)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        ax.imshow(img_small, origin="upper")
        sc_plot = ax.scatter(
            coords[order, 0],
            coords[order, 1],
            c=expr[order],
            cmap="viridis",
            s=point_size,
            alpha=0.8,
            edgecolors="none",
        )
        ax.axis("off")
        ax.set_title(f"{split} | {gene_name} (log1p)", fontsize=12)
        fig.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.02)
        save_path = out_dir / f"qc_{split}_overlay.png"
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"QC 叠加图已保存: {save_path}")
