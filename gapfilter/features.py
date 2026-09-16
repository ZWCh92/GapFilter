"""
使用预训练基础模型根据spot位置去图像中提取对应patch的特征,
保存到 data/interim/.../HE_{预训练模型名}_features.h5ad

.obsm
 |
 |-- {feature.name}_features  # (spots, feature_dim * 2) = local || contextual
 |-- spatial
"""

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import scanpy as sc
import tifffile
import timm
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def resolve_device(cfg: DictConfig) -> torch.device:
    """根据 cfg.cuda 选择设备，例如 cuda=0 -> cuda:0。"""
    cuda_id = int(OmegaConf.select(cfg, "cuda", default=0))
    if not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    if cuda_id < 0 or cuda_id >= torch.cuda.device_count():
        raise ValueError(
            f"无效 cuda={cuda_id}，当前可见 GPU 数量为 {torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{cuda_id}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(cuda_id)})")
    return device


def _seal_root(cfg) -> Path:
    """本地 clone 的 SEAL 仓库根目录。"""
    return Path(cfg.paths.pretrained_models_dir) / "SEAL" / "SEAL"


def load_seal_model(cfg, device: torch.device):
    """用官方 SEAL 流程加载 univ2 vision encoder（UNI2-h 基座 + LoRA）。"""
    seal_root = _seal_root(cfg)
    if not seal_root.is_dir():
        raise FileNotFoundError(
            f"未找到 SEAL 仓库: {seal_root}。"
            "请确认已下载到 models/pretrained_models/SEAL/SEAL"
        )

    ckpt_path = Path(cfg.feature.model_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 SEAL vision checkpoint: {ckpt_path}")

    if str(seal_root) not in sys.path:
        sys.path.insert(0, str(seal_root))

    old_cwd = Path.cwd()
    os.chdir(seal_root)
    try:
        from seal.models.load_model import ModelMixin
        from seal.utils.constants import EMB_DICT
        from seal.utils.exp_utils import update_config

        backbone = OmegaConf.select(cfg.feature, "backbone", default="univ2")
        model_config = update_config(argparse.Namespace(config="conf/config.yaml"))
        model_config.update(
            {
                "encoder": backbone,
                "partial_blocks": 3,
                "projection_head": "linear",
                "out_dim": EMB_DICT[backbone],
                "use_lora": True,
                "use_adapter": False,
            }
        )

        mixin = ModelMixin()
        mixin.conf = model_config
        mixin.emb_dict = EMB_DICT

        model, eval_transforms, precision = mixin.get_img_model(
            backbone,
            partial_blocks=model_config["partial_blocks"],
            use_adapter=False,
            adapter_bottleneck=model_config.get("adapter_bottleneck"),
            projection_head=model_config["projection_head"],
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        if any(k.startswith("module.") for k in ckpt_dict):
            ckpt_dict = {k.replace("module.", ""): v for k, v in ckpt_dict.items()}

        missing, unexpected = model.load_state_dict(ckpt_dict, strict=False)
        if unexpected:
            print(f"[SEAL] unexpected keys: {unexpected}")
        if missing:
            print(f"[SEAL] missing keys (n={len(missing)}), e.g. {missing[:5]}")

        model.to(device)
        model.eval()
        return model, eval_transforms, precision
    finally:
        os.chdir(old_cwd)


def load_h_optimus_1_model(cfg, device: torch.device):
    """从本地 HF 权重目录加载 H-optimus-1。"""
    local_dir = Path(cfg.feature.model_path)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"未找到 H-optimus-1 权重目录: {local_dir}")

    weight_file = local_dir / "model.safetensors"
    if not weight_file.exists():
        weight_file = local_dir / "pytorch_model.bin"
    if not weight_file.exists():
        raise FileNotFoundError(
            f"目录 {local_dir} 中未找到 model.safetensors 或 pytorch_model.bin"
        )

    config_file = local_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"未找到 H-optimus-1 config.json: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)

    model = timm.create_model(
        config["architecture"],
        pretrained=False,
        num_classes=0,
        img_size=config["pretrained_cfg"]["input_size"][-1],  # 224
        init_values=1e-5,
        dynamic_img_size=False,
    )
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_file))
    else:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=tuple(config["pretrained_cfg"]["mean"]),
                std=tuple(config["pretrained_cfg"]["std"]),
            ),
        ]
    )
    return model, transform, torch.float16


class DualScaleSpotDataset(Dataset):
    """按 spot 坐标截取 local / contextual patch，resize 后做 transform。"""

    def __init__(
        self,
        image,
        coordinates,
        local_read_size,
        contextual_read_size,
        transform,
        output_size=224,
    ):
        self.image = image
        self.coordinates = np.asarray(coordinates)
        self.local_read_size = max(1, int(round(local_read_size)))
        self.contextual_read_size = max(1, int(round(contextual_read_size)))
        self.transform = transform
        self.output_size = int(output_size)
        self.height, self.width = image.shape[:2]

    def __len__(self):
        return len(self.coordinates)

    def _crop_center_patch(self, center_x, center_y, read_size):
        left = int(round(center_x - read_size / 2))
        top = int(round(center_y - read_size / 2))
        right = left + read_size
        bottom = top + read_size

        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(self.width, right)
        src_bottom = min(self.height, bottom)

        crop = np.asarray(self.image[src_top:src_bottom, src_left:src_right])

        if crop.ndim == 2:
            crop = np.repeat(crop[..., None], 3, axis=-1)
        if crop.shape[-1] == 4:
            crop = crop[..., :3]
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)

        canvas = np.full((read_size, read_size, 3), 255, dtype=np.uint8)
        dst_left = src_left - left
        dst_top = src_top - top
        canvas[
            dst_top : dst_top + crop.shape[0],
            dst_left : dst_left + crop.shape[1],
        ] = crop

        patch = Image.fromarray(canvas).convert("RGB")
        return patch.resize(
            (self.output_size, self.output_size),
            resample=Image.Resampling.BICUBIC,
        )

    def __getitem__(self, index):
        center_x, center_y = self.coordinates[index]
        local_patch = self._crop_center_patch(
            center_x, center_y, self.local_read_size
        )
        contextual_patch = self._crop_center_patch(
            center_x, center_y, self.contextual_read_size
        )
        return {
            "local": self.transform(local_patch),
            "contextual": self.transform(contextual_patch),
            "index": index,
        }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg)
    feature_name = str(OmegaConf.select(cfg.feature, "name", default="") or "")
    model_path = str(cfg.feature.model_path)

    if feature_name == "h-optimus-1" or "h-optimus" in model_path.lower():
        model, transform, precision = load_h_optimus_1_model(cfg, device)
        print(f"Loaded H-optimus-1 model from {model_path}")
    elif feature_name.startswith("seal") or "seal" in model_path.lower():
        model, transform, precision = load_seal_model(cfg, device)
        print(f"Loaded SEAL model from {model_path}")
    else:
        raise ValueError(
            f"Unsupported feature model: name={feature_name}, path={model_path}"
        )

    samples = OmegaConf.select(cfg.dataset, "samples", default=None)
    if samples:
        sample_dirs = [
            os.path.join(cfg.dataset.processed_data_dir, str(s)) for s in samples
        ]
    else:
        sample_dirs = [cfg.dataset.processed_data_dir]

    use_cuda = device.type == "cuda"
    num_workers = 4 if use_cuda else 0

    for i, dataset_dir in enumerate(sample_dirs):
        sample_name = Path(dataset_dir).name

        HE_image_path = os.path.join(dataset_dir, cfg.dataset.files.image)
        if not os.path.isfile(HE_image_path):
            raise FileNotFoundError(f"未找到 HE 图像: {HE_image_path}")
        HE_image = tifffile.imread(HE_image_path)

        mpp_path = os.path.join(dataset_dir, cfg.dataset.files.mpp)
        if not os.path.isfile(mpp_path):
            raise FileNotFoundError(f"未找到 mpp: {mpp_path}")
        mpp = float(open(mpp_path).read().strip())

        adata_path = os.path.join(dataset_dir, cfg.dataset.files.expression)
        if not os.path.isfile(adata_path):
            raise FileNotFoundError(f"未找到 adata: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        print(f"Processing {sample_name}...")

        target_mpp = float(cfg.feature.target_mpp)
        patch_size_px = int(cfg.feature.patch_size_px)
        spot_diameter_um = float(cfg.dataset.metadata.spot_diameter_um)

        # local: 物理直径 spot_diameter_um 对应的像素边长
        # contextual: 在 target_mpp 下 patch_size_px 像素对应的原图像素边长
        patch_size_local = spot_diameter_um / mpp
        patch_size_contextual = patch_size_px * target_mpp / mpp

        print(
            f"mpp={mpp:.4f}, local_read={patch_size_local:.1f}px, "
            f"contextual_read={patch_size_contextual:.1f}px"
        )

        dataset = DualScaleSpotDataset(
            image=HE_image,
            coordinates=adata.obsm["spatial"],
            local_read_size=patch_size_local,
            contextual_read_size=patch_size_contextual,
            transform=transform,
            output_size=patch_size_px,
        )

        loader_kwargs = dict(
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=False,
        )
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(dataset, **loader_kwargs)

        feature_dim = int(cfg.feature.feature_dim)
        spot_features = np.empty((len(dataset), feature_dim * 2), dtype=np.float32)

        autocast_device = device.type
        autocast_dtype = precision if use_cuda else torch.float32

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Extracting [{sample_name}]", total=len(loader)):
                local_images = batch["local"]
                contextual_images = batch["contextual"]
                indices = batch["index"].numpy()
                batch_size = local_images.shape[0]

                images = torch.cat([local_images, contextual_images], dim=0).to(
                    device, non_blocking=use_cuda
                )

                if use_cuda:
                    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                        features = model(images)
                else:
                    features = model(images)

                features = features.float().cpu().numpy()
                spot_features[indices] = np.concatenate(
                    [features[:batch_size], features[batch_size:]],
                    axis=1,
                )

        feature_adata = adata.copy()
        obsm_key = f"{cfg.feature.name}_features"
        feature_adata.obsm[obsm_key] = spot_features
        feature_adata.uns["feature_extract"] = {
            "encoder": cfg.feature.name,
            "feature_dim": feature_dim,
            "layout": "local|contextual",
            "target_mpp": target_mpp,
            "patch_size_px": patch_size_px,
            "spot_diameter_um": spot_diameter_um,
            "device": str(device),
        }

        spot_feature_path = f"{dataset_dir}/expr_with_HE_SEAL_features.h5ad"
        feature_adata.write_h5ad(spot_feature_path)
        print(f"Saved features to {spot_feature_path}")


if __name__ == "__main__":
    main()
