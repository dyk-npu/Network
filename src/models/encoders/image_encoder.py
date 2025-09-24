import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm import create_model
from typing import List, Tuple, Optional, Dict
import numpy as np
import os
import warnings
import timm


class MultiViewViTEncoder(nn.Module):
    """
    多视图ViT编码器，专门处理三视图CAD零件图像
    支持前视图、侧视图、俯视图的特征提取和融合
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_views: int = 3,
        feature_dim: int = 768,
        fusion_method: str = "attention",
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: ViT模型名称
            num_views: 视图数量（默认3个：前、侧、俯）
            feature_dim: 特征维度
            fusion_method: 融合方法 ("attention", "average", "max", "concat")
            dropout: dropout率
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()

        self.num_views = num_views
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method

        # 创建ViT backbone
        self.backbone = self._create_backbone(model_name)

        # 获取backbone输出维度
        self.backbone_dim = self._get_backbone_output_dim()

        # 冻结backbone参数（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 特征投影层
        self.feature_projection = nn.Linear(self.backbone_dim, feature_dim)

        # 视图融合模块
        if fusion_method == "attention":
            self.view_attention = ViewAttention(feature_dim, num_views)
        elif fusion_method == "concat":
            self.fusion_projection = nn.Linear(feature_dim * num_views, feature_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)

    def _create_backbone(self, model_name: str):
        """创建ViT backbone"""
        try:
            # 首先尝试加载本地预训练模型
            local_model_path = r'D:\PythonProjects\Network\models\pretrained\vit_base_patch16_224\pytorch_model.bin'
            if os.path.exists(local_model_path):
                backbone = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool=""
                )
                # 加载本地权重
                state_dict = torch.load(local_model_path, map_location='cpu')
                backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded local pretrained model from {local_model_path}")
            else:
                # 回退到在线下载
                print(f"Local model not found at {local_model_path}, using online pretrained model")
                backbone = create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    global_pool=""
                )
        except Exception as e:
            print(f"Warning: Error loading pretrained model: {e}")
            print("Creating model without pretrained weights")
            backbone = create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=""
            )

        return backbone


    def _get_backbone_output_dim(self) -> int:
        """动态获取backbone的输出维度"""
        self.backbone.eval()
        with torch.no_grad():
            try:
                # 创建一个测试输入
                test_input = torch.randn(1, 3, 224, 224)
                test_output = self.backbone(test_input)

                # 处理不同的输出格式
                processed_output = self._process_backbone_output(test_output)
                output_dim = processed_output.size(1)

                print(f"Backbone output shape: {test_output.shape} -> processed: {processed_output.shape}, feature dim: {output_dim}")
                return output_dim
            except Exception as e:
                print(f"Error getting backbone output dim: {e}")
                # 根据常见的ViT模型返回默认值
                if "vit_base" in self.backbone.__class__.__name__.lower():
                    return 768
                elif "vit_large" in self.backbone.__class__.__name__.lower():
                    return 1024
                else:
                    return 768  # 默认值

    def _process_backbone_output(self, output: torch.Tensor) -> torch.Tensor:
        """统一处理backbone输出，确保返回2D张量"""
        processed_output = output

        # 处理不同维度的输出
        if len(processed_output.shape) == 4:
            # [batch, channels, height, width] -> [batch, features]
            processed_output = F.adaptive_avg_pool2d(processed_output, (1, 1))
            processed_output = processed_output.view(processed_output.size(0), -1)
        elif len(processed_output.shape) == 3:
            # [batch, sequence_length, features] -> [batch, features]
            # 对于ViT，通常取第一个token (CLS token)或者平均池化
            if processed_output.size(1) > 1:
                # 如果有多个token，取CLS token (第一个)
                processed_output = processed_output[:, 0]
            else:
                processed_output = processed_output.squeeze(1)
        elif len(processed_output.shape) == 2:
            # 已经是正确格式
            pass
        else:
            # 其他情况，展平为2D
            processed_output = processed_output.view(processed_output.size(0), -1)

        return processed_output

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: 输入图像张量 [batch_size, num_views, channels, height, width]

        Returns:
            包含各种特征表示的字典
        """
        batch_size = images.size(0)

        # 重塑为 [batch_size * num_views, channels, height, width]
        images_flat = images.view(-1, *images.shape[2:])

        # 通过backbone提取特征
        features = self.backbone(images_flat)  # 可能是 [batch_size * num_views, ...] 的各种形状

        # 使用统一的输出处理函数
        features = self._process_backbone_output(features)

        # 投影到目标维度
        features = self.feature_projection(features)  # [batch_size * num_views, feature_dim]
        features = self.dropout(features)

        # 重塑回多视图格式
        view_features = features.view(batch_size, self.num_views, self.feature_dim)

        # 视图融合
        if self.fusion_method == "attention":
            fused_features = self.view_attention(view_features)
        elif self.fusion_method == "average":
            fused_features = torch.mean(view_features, dim=1)
        elif self.fusion_method == "max":
            fused_features = torch.max(view_features, dim=1)[0]
        elif self.fusion_method == "concat":
            concat_features = view_features.view(batch_size, -1)
            fused_features = self.fusion_projection(concat_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # 层归一化
        fused_features = self.layer_norm(fused_features)

        return {
            "fused_features": fused_features,  # [batch_size, feature_dim]
            "view_features": view_features,    # [batch_size, num_views, feature_dim]
            "individual_features": {
                "front_view": view_features[:, 0],    # 前视图
                "side_view": view_features[:, 1],     # 侧视图
                "top_view": view_features[:, 2] if self.num_views > 2 else None  # 俯视图
            }
        }


class ViewAttention(nn.Module):
    """视图注意力机制，用于自适应融合多个视图的特征"""

    def __init__(self, feature_dim: int, num_views: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views

        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: [batch_size, num_views, feature_dim]

        Returns:
            融合后的特征 [batch_size, feature_dim]
        """
        # 计算每个视图的注意力权重
        attention_scores = self.attention_weights(view_features)  # [batch_size, num_views, 1]
        attention_weights = F.softmax(attention_scores, dim=1)   # [batch_size, num_views, 1]

        # 加权融合
        weighted_features = view_features * attention_weights    # [batch_size, num_views, feature_dim]
        fused_features = torch.sum(weighted_features, dim=1)     # [batch_size, feature_dim]

        return fused_features


class ImagePreprocessor:
    """图像预处理器，专门处理CAD零件的多视图图像"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # 基础变换
        basic_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]

        # 训练时的数据增强
        if augment:
            augment_transforms = [
                transforms.Resize(int(image_size[0] * 1.1)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
            self.transform = transforms.Compose(augment_transforms)
        else:
            self.transform = transforms.Compose(basic_transforms)

    def __call__(self, images: List) -> torch.Tensor:
        """
        处理多视图图像

        Args:
            images: 图像列表，包含3个视图 [front_view, side_view, top_view]

        Returns:
            处理后的图像张量 [num_views, channels, height, width]
        """
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = transforms.ToPILImage()(img)
            processed_img = self.transform(img)
            processed_images.append(processed_img)

        return torch.stack(processed_images, dim=0)


class ImageEncoder(nn.Module):
    """图像编码器的主要接口类"""

    def __init__(self, config):
        super().__init__()

        self.encoder = MultiViewViTEncoder(
            model_name=getattr(config, 'image_encoder_name', 'vit_base_patch16_224'),
            feature_dim=getattr(config, 'image_dim', 768),
            fusion_method=getattr(config, 'fusion_method', 'attention'),
            dropout=getattr(config, 'dropout', 0.1)
        )

        self.preprocessor = ImagePreprocessor(
            image_size=getattr(config, 'image_size', (224, 224)),
            augment=getattr(config, 'use_augmentation', False)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, num_views, channels, height, width]

        Returns:
            融合后的特征向量 [batch_size, feature_dim]
        """
        outputs = self.encoder(images)
        return outputs["fused_features"]

    def extract_view_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取各个视图的独立特征"""
        return self.encoder(images)


# 便捷函数
def create_image_encoder(config) -> ImageEncoder:
    """创建图像编码器的工厂函数"""
    return ImageEncoder(config)