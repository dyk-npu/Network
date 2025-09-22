import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import random


class MultiViewTransform:
    """多视图图像变换类，支持同步变换多个视图"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.augment_prob = augment_prob

        # 基础变换（始终应用）
        self.base_transforms = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        """
        应用变换到多视图图像

        Args:
            images: PIL图像列表 [front_view, side_view, top_view]

        Returns:
            变换后的图像张量 [num_views, C, H, W]
        """
        if self.augment and random.random() < self.augment_prob:
            return self._apply_augmented_transform(images)
        else:
            return self._apply_base_transform(images)

    def _apply_base_transform(self, images: List[Image.Image]) -> torch.Tensor:
        """应用基础变换"""
        transformed_images = []
        for img in images:
            transformed_img = self.base_transforms(img)
            transformed_images.append(transformed_img)
        return torch.stack(transformed_images, dim=0)

    def _apply_augmented_transform(self, images: List[Image.Image]) -> torch.Tensor:
        """应用增强变换，保持多视图的一致性"""
        # 为所有视图生成相同的随机参数
        augment_params = self._generate_augment_params()

        transformed_images = []
        for img in images:
            # 应用一致的增强
            augmented_img = self._apply_consistent_augment(img, augment_params)
            transformed_images.append(augmented_img)

        return torch.stack(transformed_images, dim=0)

    def _generate_augment_params(self) -> dict:
        """生成增强参数"""
        return {
            "rotation": random.uniform(-15, 15),
            "scale": random.uniform(0.9, 1.1),
            "brightness": random.uniform(0.8, 1.2),
            "contrast": random.uniform(0.8, 1.2),
            "saturation": random.uniform(0.8, 1.2),
            "hue": random.uniform(-0.1, 0.1),
            "flip_horizontal": random.random() > 0.5,
            "crop_params": T.RandomResizedCrop.get_params(
                Image.new("RGB", (256, 256)),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            )
        }

    def _apply_consistent_augment(self, img: Image.Image, params: dict) -> torch.Tensor:
        """应用一致的增强变换"""
        # 颜色增强
        img = TF.adjust_brightness(img, params["brightness"])
        img = TF.adjust_contrast(img, params["contrast"])
        img = TF.adjust_saturation(img, params["saturation"])
        img = TF.adjust_hue(img, params["hue"])

        # 几何变换
        if params["flip_horizontal"]:
            img = TF.hflip(img)

        # 旋转
        img = TF.rotate(img, params["rotation"])

        # 随机裁剪和缩放
        i, j, h, w = params["crop_params"]
        img = TF.resized_crop(img, i, j, h, w, self.image_size)

        # 转换为张量并归一化
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        return img


class CADAugmentation:
    """CAD零件专用的数据增强"""

    def __init__(
        self,
        geometric_prob: float = 0.5,
        photometric_prob: float = 0.5,
        noise_prob: float = 0.3
    ):
        self.geometric_prob = geometric_prob
        self.photometric_prob = photometric_prob
        self.noise_prob = noise_prob

    def __call__(self, images: List[Image.Image]) -> List[Image.Image]:
        """应用CAD专用增强"""
        augmented_images = []

        for img in images:
            # 几何增强
            if random.random() < self.geometric_prob:
                img = self._apply_geometric_augment(img)

            # 光度增强
            if random.random() < self.photometric_prob:
                img = self._apply_photometric_augment(img)

            # 噪声增强
            if random.random() < self.noise_prob:
                img = self._apply_noise_augment(img)

            augmented_images.append(img)

        return augmented_images

    def _apply_geometric_augment(self, img: Image.Image) -> Image.Image:
        """几何增强（保持CAD特征）"""
        # 轻微旋转（CAD图像通常需要保持方向）
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle, fill=255)

        # 轻微缩放
        scale = random.uniform(0.95, 1.05)
        w, h = img.size
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

        return img

    def _apply_photometric_augment(self, img: Image.Image) -> Image.Image:
        """光度增强"""
        # 对比度调整
        contrast_factor = random.uniform(0.9, 1.1)
        img = TF.adjust_contrast(img, contrast_factor)

        # 亮度调整
        brightness_factor = random.uniform(0.9, 1.1)
        img = TF.adjust_brightness(img, brightness_factor)

        return img

    def _apply_noise_augment(self, img: Image.Image) -> Image.Image:
        """噪声增强"""
        img_array = np.array(img)

        # 高斯噪声
        noise = np.random.normal(0, 5, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)


class ViewSpecificTransform:
    """视图特定的变换，为不同视图应用不同的增强策略"""

    def __init__(
        self,
        base_transform: MultiViewTransform,
        view_configs: Optional[dict] = None
    ):
        self.base_transform = base_transform
        self.view_configs = view_configs or {
            "front": {"rotation_range": (-10, 10)},
            "side": {"rotation_range": (-5, 5)},
            "top": {"rotation_range": (-15, 15)}
        }

    def __call__(self, images: List[Image.Image], view_names: List[str]) -> torch.Tensor:
        """
        为不同视图应用特定变换

        Args:
            images: 图像列表
            view_names: 视图名称列表

        Returns:
            变换后的图像张量
        """
        transformed_images = []

        for img, view_name in zip(images, view_names):
            if view_name in self.view_configs:
                config = self.view_configs[view_name]
                img = self._apply_view_specific_augment(img, config)

            # 应用基础变换
            img_tensor = T.Compose([
                T.Resize(self.base_transform.image_size),
                T.ToTensor(),
                T.Normalize(mean=self.base_transform.mean, std=self.base_transform.std)
            ])(img)

            transformed_images.append(img_tensor)

        return torch.stack(transformed_images, dim=0)

    def _apply_view_specific_augment(self, img: Image.Image, config: dict) -> Image.Image:
        """应用视图特定的增强"""
        if "rotation_range" in config:
            min_angle, max_angle = config["rotation_range"]
            angle = random.uniform(min_angle, max_angle)
            img = TF.rotate(img, angle, fill=255)

        return img


def create_train_transform(
    image_size: Tuple[int, int] = (224, 224),
    augment: bool = True
) -> MultiViewTransform:
    """创建训练时的变换"""
    return MultiViewTransform(
        image_size=image_size,
        augment=augment,
        augment_prob=0.5
    )


def create_val_transform(
    image_size: Tuple[int, int] = (224, 224)
) -> MultiViewTransform:
    """创建验证时的变换"""
    return MultiViewTransform(
        image_size=image_size,
        augment=False
    )


def create_test_transform(
    image_size: Tuple[int, int] = (224, 224)
) -> MultiViewTransform:
    """创建测试时的变换"""
    return MultiViewTransform(
        image_size=image_size,
        augment=False
    )