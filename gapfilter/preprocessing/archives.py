"""压缩包解压工具。

``data/raw`` 中的 VisiumHD/Xenium 数据以 ``*.tar.gz`` / ``*.zip`` 形式存在，属于运算量
大的中间处理，统一解压到 ``data/interim``，绝不写回 ``data/raw``。

解压是幂等的：若目标目录中已存在“完成标记”文件，则跳过重复解压。
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Optional

from loguru import logger

_DONE_SUFFIX = ".extracted.done"


def _done_marker(dest_dir: Path, archive: Path) -> Path:
    return dest_dir / f"{archive.name}{_DONE_SUFFIX}"


def extract_archive(
    archive: Path, dest_dir: Path, overwrite: bool = False
) -> Path:
    """将 ``archive`` 解压到 ``dest_dir``（幂等）。返回 ``dest_dir``。

    支持 ``.tar.gz`` / ``.tgz`` / ``.tar`` / ``.zip``。已解压过（存在完成标记）则跳过。
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = _done_marker(dest_dir, archive)

    if marker.exists() and not overwrite:
        logger.info(f"已解压，跳过: {archive.name} -> {dest_dir}")
        return dest_dir

    logger.info(f"解压 {archive} -> {dest_dir}")
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if name.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive, mode) as tar:
            _safe_extract_tar(tar, dest_dir)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {archive}")

    marker.touch()
    logger.success(f"解压完成: {archive.name}")
    return dest_dir


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    for member in tar.getmembers():
        member_path = dest_dir / member.name
        if not _is_within(dest_dir, member_path):
            raise RuntimeError(f"检测到不安全的 tar 路径，已中止: {member.name}")
    tar.extractall(dest_dir)


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    for name in zf.namelist():
        member_path = dest_dir / name
        if not _is_within(dest_dir, member_path):
            raise RuntimeError(f"检测到不安全的 zip 路径，已中止: {name}")
    zf.extractall(dest_dir)


def find_one(root: Path, pattern: str, max_depth: int = 8) -> Optional[Path]:
    """在 ``root`` 下有界深度查找第一个匹配 ``pattern`` 的文件，找不到返回 ``None``。

    ``pattern`` 可含 ``/`` 表示相对多级路径；使用有界深度避免在慢速盘上深度递归。
    """
    root = Path(root)
    for depth in range(max_depth + 1):
        prefix = "/".join(["*"] * depth)
        glob_pattern = f"{prefix}/{pattern}" if prefix else pattern
        matches = sorted(root.glob(glob_pattern))
        if matches:
            return matches[0]
    return None
