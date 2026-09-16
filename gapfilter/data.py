"""空间转录组数据预处理 CLI 入口。

根据 ``RAW_DATA_DIR`` 下不同数据类型（Visium / VisiumHD / Xenium）采取不同预处理策略：

- 运算量大的中间结果（解压后的原始文件、聚合后的 AnnData）写入 ``data/interim``；
- 预处理后的最终结果写入 ``data/processed``；
- ``data/raw`` 始终只读；
- 支持传入任意文件夹路径，只处理该路径下的样本，实现模块解耦。

预处理每个样本产出：

``data/processed/<类型>/<样本>/``：

1. ``HE.tif``：H&E 图像；``MPP.txt``：每像素微米数 (microns-per-pixel)；
2. ``SVG_top_200.txt`` / ``SVG_top_2000.txt``：从 HEG∩HVG top-5000 候选池中按 Moran's I
   从高到低排序的空间可变基因清单（候选池基因数不足 2000 时退化为 ``SVG_top_{num}.txt``）；
3. ``expr.h5ad``：子集到 ``min_cells`` 过滤后基因的 AnnData，``X`` 为原始计数，
   ``layers['log1p_cpm']`` 为归一化表达，``obs['split']`` 为 train/test 划分，
   ``obsm['spatial']`` 为 H&E 像素坐标。

``data/interim/<类型>/<样本>/``（运算量大的中间产物）：

- 解压后的原始文件、聚合/对齐后的 AnnData（续跑缓存）；
- ``moranI.csv``：候选池的 Moran's I 结果；
- ``qc_train_overlay.png`` / ``qc_test_overlay.png``：train/test spot 叠加 H&E 的 QC 图。

用法示例::

    # 从 raw 全量处理
    python gapfilter/data.py --overwrite

    # 从 interim 聚合缓存续跑（跳过解压/聚合，重做基因选择与 expr.h5ad）
    python gapfilter/data.py --from-interim --overwrite

    # 只处理某个文件夹
    python gapfilter/data.py -i data/raw/VisiumHD --from-interim --overwrite
"""

from pathlib import Path
from typing import Optional

from loguru import logger
import typer

from config import RAW_DATA_DIR
from preprocessing.config import PreprocessConfig
from preprocessing.registry import process_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Option(
        RAW_DATA_DIR,
        "--input-path",
        "-i",
        help="待处理的文件夹路径（raw 根、某类型目录或单个样本目录均可）。",
    ),
    data_type: Optional[str] = typer.Option(
        None,
        "--data-type",
        "-t",
        help="显式指定数据类型 Visium/VisiumHD/Xenium（默认自动识别）。",
    ),
    visiumhd_bin: str = typer.Option(
        "square_008um", help="VisiumHD 使用的 bin 分辨率层级。"
    ),
    xenium_qv: float = typer.Option(20.0, help="Xenium 转录本 QV 过滤阈值。"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="若最终结果已存在则重新处理。"
    ),
    from_interim: bool = typer.Option(
        False,
        "--from-interim",
        help="从 interim 聚合/对齐缓存续跑（跳过 raw 解压与聚合）；需已有对应 h5ad。",
    ),
    no_log1p_layer: bool = typer.Option(
        False, "--no-log1p-layer", help="不额外保存 log1p-CPM 层。"
    ),
):
    """预处理 ``input_path`` 下的空间转录组数据。"""
    config = PreprocessConfig(
        visiumhd_bin=visiumhd_bin,
        xenium_qv_threshold=xenium_qv,
        overwrite=overwrite,
        from_interim=from_interim,
        save_log1p_cpm_layer=not no_log1p_layer,
    )

    mode = "interim 续跑" if from_interim else "raw 全量"
    logger.info(f"开始预处理 ({mode}): {input_path}")
    result = process_path(input_path, config=config, data_type=data_type)

    if result["skipped"]:
        logger.warning("以下样本被跳过：")
        for item in result["skipped"]:
            logger.warning(f"  - {item}")
    if result["failed"]:
        logger.error("以下样本处理失败：")
        for item in result["failed"]:
            logger.error(f"  - {item}")

    logger.success("全部预处理流程结束。")


if __name__ == "__main__":
    app()
