"""预处理基类与样本描述。

每种数据类型的处理器继承 :class:`BaseProcessor`，只需实现 :meth:`process_sample`。
样本的路径信息统一由 :class:`SampleSpec` 描述，实现原始/中间/结果三类目录的解耦。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from loguru import logger

from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from preprocessing.config import PreprocessConfig


class PreprocessError(RuntimeError):
    """预处理过程中可预期的错误（如缺少必要文件），用于优雅跳过。"""


@dataclass
class SampleSpec:
    """单个待处理样本的描述。

    属性
    ----
    data_type : str
        ``Visium`` / ``VisiumHD`` / ``Xenium`` 之一。
    name : str
        样本名称（相对数据类型根目录的路径，用于命名输出，如 ``DLPFC/151507``）。
    raw_dir : Path
        样本原始数据所在目录（只读）。
    files : Dict[str, Path]
        已解析出的关键原始文件路径（如 he、counts、positions 等）。
    """

    data_type: str
    name: str
    raw_dir: Path
    files: Dict[str, Path] = field(default_factory=dict)

    @property
    def safe_name(self) -> str:
        """可用于文件系统的名称（把分隔符替换为下划线）。"""
        return self.name.replace("/", "__")

    @property
    def interim_dir(self) -> Path:
        return INTERIM_DATA_DIR / self.data_type / self.name

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DATA_DIR / self.data_type / self.name

    @property
    def output_h5ad(self) -> Path:
        return self.processed_dir / "expr.h5ad"


class BaseProcessor(abc.ABC):
    """预处理器基类。"""

    data_type: str = "Base"

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def run(self, sample: SampleSpec) -> bool:
        """执行单个样本的完整预处理。返回是否成功。"""
        if sample.output_h5ad.exists() and not self.config.overwrite:
            logger.info(
                f"[{sample.data_type}] 结果已存在，跳过 {sample.name} "
                f"(使用 --overwrite 可重跑): {sample.output_h5ad}"
            )
            return True

        assert self._is_under_raw(sample.raw_dir), "样本目录必须位于 data/raw 之下"
        sample.interim_dir.mkdir(parents=True, exist_ok=True)
        sample.processed_dir.mkdir(parents=True, exist_ok=True)

        mode = "from-interim" if self.config.from_interim else "from-raw"
        logger.info(
            f"===== 开始处理 [{sample.data_type}] {sample.name} ({mode}) ====="
        )
        self.process_sample(sample)
        logger.success(f"===== 完成 [{sample.data_type}] {sample.name} =====")
        return True

    @staticmethod
    def _is_under_raw(path: Path) -> bool:
        try:
            path.resolve().relative_to(RAW_DATA_DIR.resolve())
            return True
        except ValueError:
            return False

    @abc.abstractmethod
    def process_sample(self, sample: SampleSpec) -> None:
        """子类实现具体的预处理流程。"""
        raise NotImplementedError
