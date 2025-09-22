import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """数据配置类"""

    # 数据路径配置
    raw_data_dir: str = os.path.join(BaseConfig.data_root, "raw")
    processed_data_dir: str = os.path.join(BaseConfig.data_root, "processed")
    annotation_dir: str = os.path.join(BaseConfig.data_root, "annotations")

    # 数据集划分
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # 图像预处理配置
    image_size: Tuple[int, int] = (224, 224)
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # 文本预处理配置
    max_text_length: int = 512
    text_tokenizer: str = "bert-base-uncased"

    # Brep模型预处理配置
    brep_sample_method: str = "uv_sampling"
    brep_resolution: int = 64
    brep_normalize: bool = True

    # 数据加载配置
    pin_memory: bool = True
    prefetch_factor: int = 2

    # 支持的文件格式
    supported_image_formats: List[str] = None
    supported_brep_formats: List[str] = None

    def __post_init__(self):
        super().__post_init__()

        if self.supported_image_formats is None:
            self.supported_image_formats = [".jpg", ".jpeg", ".png", ".bmp"]

        if self.supported_brep_formats is None:
            self.supported_brep_formats = [".step", ".stp", ".iges", ".igs", ".brep"]

        # 创建数据目录
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)


@dataclass
class AugmentationConfig:
    """数据增强配置类"""

    # 图像增强
    random_rotation: bool = True
    rotation_range: Tuple[int, int] = (-30, 30)

    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)

    random_flip: bool = True
    flip_probability: float = 0.5

    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1

    # 文本增强
    text_synonym_replacement: bool = True
    synonym_prob: float = 0.1

    text_random_insertion: bool = True
    insertion_prob: float = 0.1

    # Brep模型增强
    brep_random_rotation: bool = True
    brep_noise_std: float = 0.01
    brep_scaling_range: Tuple[float, float] = (0.9, 1.1)