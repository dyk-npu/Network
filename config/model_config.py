from dataclasses import dataclass
from typing import Dict, List
from .base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """模型配置类"""

    # 编码器配置
    text_encoder_name: str = "bert-base-uncased"
    image_encoder_name: str = "vit_base_patch16_224"

    # 特征维度
    text_dim: int = 768
    image_dim: int = 768
    brep_dim: int = 1024
    fusion_dim: int = 512

    # Brep编码器配置
    brep_sample_points: int = 2048
    brep_hidden_dims: List[int] = None

    # 融合模块配置
    num_attention_heads: int = 8
    num_transformer_layers: int = 6
    dropout: float = 0.1

    # 下游任务配置
    num_classes: int = 12  # 分类任务类别数
    feature_types: List[str] = None  # 特征识别类型

    # 损失函数权重
    classification_weight: float = 1.0
    retrieval_weight: float = 1.0
    feature_detection_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        if self.brep_hidden_dims is None:
            self.brep_hidden_dims = [512, 256, 128]

        if self.feature_types is None:
            self.feature_types = ["hole", "slot", "chamfer", "fillet", "plane", "cylinder"]


@dataclass
class TrainingConfig(BaseConfig):
    """训练配置类"""

    # 优化器配置
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 10

    # 损失函数配置
    classification_loss: str = "cross_entropy"
    retrieval_loss: str = "contrastive"
    feature_detection_loss: str = "focal"

    # 数据增强
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # 正则化
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2