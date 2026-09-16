"""空间转录组数据预处理子包。

按数据类型（Visium / VisiumHD / Xenium）对 ``data/raw`` 下的数据采取不同的预处理
策略，实现模块解耦：

- 传入任意文件夹路径，自动识别该路径下的数据类型并只处理该路径下的样本；
- 运算量大的中间结果（解压后的原始文件、聚合后的 AnnData）写入 ``data/interim``；
- 预处理后的最终结果写入 ``data/processed``；
- ``data/raw`` 始终只读，不被修改。

对外主要入口：

- :func:`preprocessing.registry.process_path` —— 处理某个路径下的所有样本；
- :class:`preprocessing.base.SampleSpec` —— 单个样本的描述；
- 各数据类型的处理器：:class:`VisiumProcessor` / :class:`VisiumHDProcessor` /
  :class:`XeniumProcessor`。
"""

from preprocessing.base import BaseProcessor, PreprocessError, SampleSpec
from preprocessing.config import PreprocessConfig
from preprocessing.registry import (
    detect_data_type,
    discover_samples,
    get_processor,
    process_path,
)
from preprocessing.visium import VisiumProcessor
from preprocessing.visiumhd import VisiumHDProcessor
from preprocessing.xenium import XeniumProcessor

__all__ = [
    "BaseProcessor",
    "PreprocessConfig",
    "PreprocessError",
    "SampleSpec",
    "VisiumProcessor",
    "VisiumHDProcessor",
    "XeniumProcessor",
    "detect_data_type",
    "discover_samples",
    "get_processor",
    "process_path",
]
