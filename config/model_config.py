from dataclasses import dataclass
from typing import Dict, List
from .base_config import BaseConfig

# CADNET数据集的标准分类定义（43类）
CAD_PART_CLASSES = [
    "90_degree_elbows", "BackDoors", "Bearing_Blocks", "Bearing_Like_Parts",
    "Bolt_Like_Parts", "Bracket_like_Parts", "Clips", "Contact_Switches",
    "Container_Like_Parts", "Contoured_Surfaces", "Curved_Housings", "Cylindrical_Parts",
    "Discs", "Flange_Like_Parts", "Gear_like_Parts", "Handles",
    "Intersecting_Pipes", "L_Blocks", "Long_Machine_Elements", "Long_Pins",
    "Machined_Blocks", "Machined_Plates", "Motor_Bodies", "Non-90_degree_elbows",
    "Nuts", "Oil_Pans", "Posts", "Prismatic_Stock",
    "Pulley_Like_Parts", "Rectangular_Housings", "Rocker_Arms", "Round_Change_At_End",
    "Screw", "Simple_Pipes", "Slender_Links", "Slender_Thin_Plates",
    "Small_Machined_Blocks", "Spoked_Wheels", "Springs", "Thick_Plates",
    "Thin_Plates", "T-shaped_parts", "U-shaped_parts"
]

# 特征检测类型定义
FEATURE_DETECTION_TYPES = [
    "hole", "slot", "chamfer", "fillet", "plane", "cylinder"
]

VIT_MODEL_LOCAL_PATH = r'D:\PythonProjects\Network\models\pretrained\vit_base_patch16_224\pytorch_model.bin'


def get_category_mapping() -> Dict[str, int]:
    """获取类别名称到索引的映射"""
    return {category: idx for idx, category in enumerate(CAD_PART_CLASSES)}


def get_index_to_category() -> Dict[int, str]:
    """获取索引到类别名称的映射"""
    return {idx: category for idx, category in enumerate(CAD_PART_CLASSES)}


def validate_category(category: str) -> bool:
    """验证类别是否有效"""
    return category in CAD_PART_CLASSES


def validate_label(label: int) -> bool:
    """验证标签索引是否有效"""
    return 0 <= label < len(CAD_PART_CLASSES)


def category_to_label(category: str) -> int:
    """将类别字符串转换为数字标签"""
    mapping = get_category_mapping()
    if category not in mapping:
        raise ValueError(f"Unknown category: {category}. Valid categories: {CAD_PART_CLASSES}")
    return mapping[category]


def label_to_category(label: int) -> str:
    """将数字标签转换为类别字符串"""
    if not validate_label(label):
        raise ValueError(f"Invalid label: {label}. Valid range: 0-{len(CAD_PART_CLASSES)-1}")
    return CAD_PART_CLASSES[label]


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
    num_classes: int = len(CAD_PART_CLASSES)  # CADNET数据集类别数，动态计算
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
            self.feature_types = FEATURE_DETECTION_TYPES.copy()


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