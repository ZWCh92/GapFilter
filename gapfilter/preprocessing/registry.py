"""数据类型识别、样本发现与分发。

对外主入口 :func:`process_path`：给定任意路径，自动识别其下的数据类型并只处理该路径
下的样本，天然实现“输入一个文件夹路径即可只处理该文件夹数据”的模块解耦需求。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from preprocessing.base import BaseProcessor, PreprocessError, SampleSpec
from preprocessing.config import PreprocessConfig
from preprocessing.visium import VisiumProcessor
from preprocessing.visiumhd import VisiumHDProcessor
from preprocessing.xenium import XeniumProcessor

# 注意顺序：VisiumHD 必须在 Visium 之前判断（名称包含 "Visium" 子串）
DATA_TYPES = ("VisiumHD", "Xenium", "Visium")

PROCESSORS: Dict[str, Type[BaseProcessor]] = {
    "Visium": VisiumProcessor,
    "VisiumHD": VisiumHDProcessor,
    "Xenium": XeniumProcessor,
}


def get_processor(data_type: str, config: PreprocessConfig | None = None) -> BaseProcessor:
    if data_type not in PROCESSORS:
        raise ValueError(f"未知数据类型: {data_type}")
    return PROCESSORS[data_type](config)


# ---------------------------------------------------------------------------
# 类型识别
# ---------------------------------------------------------------------------


def detect_data_type(path: Path) -> Optional[str]:
    """识别路径对应的数据类型：先看路径中的类型目录名，再看目录内容特征。"""
    by_parts = _detect_by_parts(path)
    if by_parts is not None:
        return by_parts
    return _detect_by_content(path)


def _detect_by_parts(path: Path) -> Optional[str]:
    """仅依据路径中是否包含类型目录名来判断（不看内容，避免根目录被误判）。"""
    parts = set(Path(path).resolve().parts)
    for dtype in DATA_TYPES:
        if dtype in parts:
            return dtype
    return None


def _detect_by_content(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    if _bounded_find(path, "*_outs.zip") or _bounded_find(path, "experiment.xenium"):
        return "Xenium"
    if _bounded_find(path, "*_binned_outputs.tar.gz") or _bounded_find(
        path, "binned_outputs"
    ):
        return "VisiumHD"
    if _bounded_find(path, "filtered_feature_bc_matrix.h5"):
        return "Visium"
    return None


def _type_root(sample_dir: Path, dtype: str) -> Path:
    """返回路径中名为 dtype 的祖先目录（用于计算样本相对名）。"""
    sample_dir = sample_dir.resolve()
    for p in [sample_dir, *sample_dir.parents]:
        if p.name == dtype:
            return p
    return sample_dir


def _rel_name(sample_dir: Path, dtype: str) -> str:
    root = _type_root(sample_dir, dtype)
    sample_dir = sample_dir.resolve()
    if sample_dir == root:
        return sample_dir.name
    try:
        return str(sample_dir.relative_to(root))
    except ValueError:
        return sample_dir.name


# ---------------------------------------------------------------------------
# 样本发现
# ---------------------------------------------------------------------------


def discover_samples(path: Path, config: PreprocessConfig | None = None) -> List[SampleSpec]:
    """发现 ``path`` 下所有可处理的样本。"""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    # 1) 路径本身位于某个类型目录之下（类型目录或单样本目录）：只发现该类型
    dtype = _detect_by_parts(path)
    if dtype is not None:
        return _discover_for_type(path, dtype)

    # 2) 路径是包含多种类型的容器目录（如 data/raw 根）：三种类型各扫一遍并合并。
    #    每个发现函数都使用精确的签名 glob，因此不会互相误配。
    samples: List[SampleSpec] = []
    for dt in DATA_TYPES:
        samples.extend(_discover_for_type(path, dt))
    return samples


def _discover_for_type(path: Path, dtype: str) -> List[SampleSpec]:
    if dtype == "Xenium":
        return _discover_xenium(path)
    if dtype == "VisiumHD":
        return _discover_visiumhd(path)
    if dtype == "Visium":
        return _discover_visium(path)
    return []


# 数据集在 data/raw 下的层级很浅（类型/样本/[子目录]/文件），因此用有界深度的 glob
# 代替 rglob，避免在慢速网络盘上递归遍历导致长时间卡顿。
_MAX_SEARCH_DEPTH = 5


def _bounded_find(base: Path, filename: str, max_depth: int = _MAX_SEARCH_DEPTH) -> List[Path]:
    """在 ``base`` 下有界深度地查找匹配 ``filename`` 的文件（含 base 自身层）。"""
    base = Path(base)
    results: List[Path] = []
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + [filename]) if depth else filename
        results.extend(base.glob(pattern))
    return sorted(set(results))


def _sample_dirs_by_signature(path: Path, signature_glob: str) -> List[Path]:
    matches = _bounded_find(path, signature_glob)
    return sorted({m.parent for m in matches})


def _discover_xenium(path: Path) -> List[SampleSpec]:
    samples: List[SampleSpec] = []
    for sdir in _sample_dirs_by_signature(path, "*_outs.zip"):
        he = _first(sdir, ["*_he_image.ome.tif", "*_he_image.tif", "*he*.ome.tif"])
        files = {
            "outs_zip": _first(sdir, ["*_outs.zip"]),
            "he": he,
            "alignment": _first(sdir, ["*he_imagealignment.csv", "*alignment*.csv"]),
        }
        samples.append(
            SampleSpec("Xenium", _rel_name(sdir, "Xenium"), sdir, _clean(files))
        )
    return samples


def _discover_visiumhd(path: Path) -> List[SampleSpec]:
    samples: List[SampleSpec] = []
    for sdir in _sample_dirs_by_signature(path, "*_binned_outputs.tar.gz"):
        files = {
            "binned_tar": _first(sdir, ["*_binned_outputs.tar.gz"]),
            "he": _first(sdir, ["*_tissue_image.tif", "*tissue_image*.btf", "*.btf"]),
        }
        samples.append(
            SampleSpec("VisiumHD", _rel_name(sdir, "VisiumHD"), sdir, _clean(files))
        )
    return samples


def _discover_visium(path: Path) -> List[SampleSpec]:
    """支持两种布局：DLPFC 风格（h5/ + images/ 多样本）与标准 spaceranger 单样本。"""
    path = Path(path)
    samples: List[SampleSpec] = []

    # DLPFC 风格：某目录同时含 h5/ 与 images/
    dlpfc_dirs = sorted(
        {p.parent for p in _bounded_find(path, "h5") if (p.parent / "images").is_dir()}
        | ({path} if (path / "h5").is_dir() and (path / "images").is_dir() else set())
    )
    for base in dlpfc_dirs:
        for h5 in sorted((base / "h5").glob("*_filtered_feature_bc_matrix.h5")):
            sid = h5.name.replace("_filtered_feature_bc_matrix.h5", "")
            image = _first(base / "images", [f"{sid}_full_image.tif", f"{sid}*.tif"])
            positions = _first(
                base,
                [
                    f"metadata/{sid}/tissue_positions_list.txt",
                    f"metadata/{sid}/tissue_positions_list.csv",
                    f"metadata/{sid}/tissue_positions.csv",
                    f"metadata/{sid}/tissue_positions.parquet",
                    f"spatial/{sid}/tissue_positions*.csv",
                    f"*{sid}*positions*.csv",
                ],
            )
            scalefactors = _first(base, [f"metadata/{sid}/scalefactors_json.json"])
            files = _clean(
                {
                    "counts": h5,
                    "he": image,
                    "positions": positions,
                    "scalefactors": scalefactors,
                }
            )
            name = f"{_rel_name(base, 'Visium')}/{sid}"
            samples.append(SampleSpec("Visium", name, base, files))

    # 标准 spaceranger 布局：filtered_feature_bc_matrix.h5 + spatial/
    for h5 in _bounded_find(path, "filtered_feature_bc_matrix.h5"):
        sdir = h5.parent
        if sdir.name == "h5":  # 已由 DLPFC 分支处理
            continue
        positions = _first(
            sdir,
            [
                "spatial/tissue_positions_list.csv",
                "spatial/tissue_positions.csv",
                "spatial/tissue_positions.parquet",
            ],
        )
        files = _clean(
            {
                "counts": h5,
                "he": _first(sdir, ["*.tif", "spatial/*.tif"]),
                "positions": positions,
                "scalefactors": _first(sdir, ["spatial/scalefactors_json.json"]),
            }
        )
        samples.append(SampleSpec("Visium", _rel_name(sdir, "Visium"), sdir, files))

    return samples


def _first(base: Path, patterns: List[str]) -> Optional[Path]:
    base = Path(base)
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if not matches and "/" not in pat:
            matches = _bounded_find(base, pat, max_depth=3)
        if matches:
            return matches[0]
    return None


def _clean(files: Dict[str, Optional[Path]]) -> Dict[str, Path]:
    return {k: v for k, v in files.items() if v is not None}


# ---------------------------------------------------------------------------
# 处理入口
# ---------------------------------------------------------------------------


def process_path(
    path: Path,
    config: PreprocessConfig | None = None,
    data_type: Optional[str] = None,
) -> Dict[str, List[str]]:
    """处理 ``path`` 下的所有样本，返回 ``{"success": [...], "skipped": [...], "failed": [...]}``。

    参数
    ----
    path : Path
        待处理的文件夹路径（可以是 raw 根、某类型目录或单个样本目录）。
    data_type : Optional[str]
        显式指定数据类型（可选）；不指定时自动识别。
    """
    config = config or PreprocessConfig()
    samples = discover_samples(path, config)
    if data_type is not None:
        samples = [s for s in samples if s.data_type == data_type]

    if not samples:
        logger.warning(f"在 {path} 下未发现可处理的样本。")
        return {"success": [], "skipped": [], "failed": []}

    logger.info(f"共发现 {len(samples)} 个样本待处理:")
    for s in samples:
        logger.info(f"  - [{s.data_type}] {s.name}  ({s.raw_dir})")

    result: Dict[str, List[str]] = {"success": [], "skipped": [], "failed": []}
    for sample in samples:
        tag = f"[{sample.data_type}] {sample.name}"
        try:
            processor = get_processor(sample.data_type, config)
            processor.run(sample)
            result["success"].append(tag)
        except PreprocessError as e:
            logger.warning(f"跳过 {tag}: {e}")
            result["skipped"].append(f"{tag}: {e}")
        except Exception as e:  # noqa: BLE001 - 单样本失败不应中断整体流程
            logger.exception(f"处理失败 {tag}: {e}")
            result["failed"].append(f"{tag}: {e}")

    logger.info(
        f"完成。成功 {len(result['success'])}，跳过 {len(result['skipped'])}，"
        f"失败 {len(result['failed'])}。"
    )
    return result
