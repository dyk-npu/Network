import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MultiViewCADDataset(Dataset):
    """
    多视图CAD零件数据集
    支持加载前视图、侧视图、俯视图的图像数据
    适配文件夹结构: model_name/front_view.png, model_name/side_view.png, model_name/top_view.png
    """

    def __init__(
        self,
        data_dir: str,
        annotation_file: Optional[str] = None,
        transform=None,
        views: List[str] = ["front", "side", "top"],
        view_files: Dict[str, str] = None
    ):
        """
        Args:
            data_dir: 图像数据根目录，包含以模型名称命名的文件夹
            annotation_file: 标注文件路径（JSON格式），可选
            transform: 图像变换函数
            views: 视图名称列表
            view_files: 视图文件名映射 {view_name: filename}
        """
        self.data_dir = Path(data_dir)
        self.views = views
        self.transform = transform

        # 默认视图文件映射
        if view_files is None:
            self.view_files = {
                "front": "front_view.png",
                "side": "side_view.png",
                "top": "top_view.png"
            }
        else:
            self.view_files = view_files

        # 加载标注文件（如果提供）
        self.annotations = self._load_annotations(annotation_file) if annotation_file else {}
        self.samples = self._prepare_samples()

        logger.info(f"Loaded {len(self.samples)} samples with {len(views)} views each")

    def _load_annotations(self, annotation_file: str) -> Dict:
        """加载标注文件"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        return annotations

    def _prepare_samples(self) -> List[Dict]:
        """准备样本列表"""
        samples = []

        # 扫描数据目录中的所有模型文件夹
        model_folders = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for model_folder in model_folders:
            model_name = model_folder.name
            sample = {
                "item_id": model_name,
                "views": {},
                "metadata": self.annotations.get(model_name, {}).get("metadata", {})
            }

            # 检查每个视图的图像文件
            valid_views = 0
            for view in self.views:
                view_filename = self.view_files.get(view)
                if view_filename:
                    image_path = model_folder / view_filename
                    if image_path.exists():
                        sample["views"][view] = image_path
                        valid_views += 1
                    else:
                        logger.warning(f"Missing {view} view file {view_filename} for model {model_name}")

            # 只保留有足够视图的样本
            if valid_views >= len(self.views):
                samples.append(sample)
            else:
                logger.warning(f"Skipping model {model_name}: only {valid_views}/{len(self.views)} views available")

        return samples


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        获取数据样本

        Returns:
            Dict包含:
            - images: 多视图图像张量 [num_views, C, H, W]
            - item_id: 零件ID
            - metadata: 元数据
        """
        sample = self.samples[idx]

        # 加载多视图图像
        images = []
        for view in self.views:
            if view in sample["views"]:
                image_path = sample["views"][view]
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            else:
                # 如果缺少某个视图，创建空白图像
                images.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))

        # 应用变换
        if self.transform:
            images = self.transform(images)
        else:
            # 默认变换
            images = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images])

        return {
            "images": images,
            "item_id": sample["item_id"],
            "metadata": sample["metadata"]
        }


class CADImageDataLoader:
    """CAD图像数据加载器工具类"""

    @staticmethod
    def create_annotation_template(
        image_dir: str,
        output_file: str,
        views: List[str] = ["front", "side", "top"],
        view_files: Dict[str, str] = None
    ):
        """
        创建标注文件模板
        适配文件夹结构: model_name/front_view.png, model_name/side_view.png, model_name/top_view.png

        Args:
            image_dir: 图像目录，包含以模型名称命名的子文件夹
            output_file: 输出标注文件路径
            views: 视图列表
            view_files: 视图文件名映射
        """
        image_dir = Path(image_dir)
        annotations = {}

        # 默认视图文件映射
        if view_files is None:
            view_files = {
                "front": "front_view.png",
                "side": "side_view.png",
                "top": "top_view.png"
            }

        # 扫描所有模型文件夹
        model_folders = [d for d in image_dir.iterdir() if d.is_dir()]

        for model_folder in model_folders:
            model_name = model_folder.name

            # 检查是否有所需的视图文件
            available_views = {}
            for view in views:
                view_filename = view_files.get(view)
                if view_filename:
                    view_path = model_folder / view_filename
                    if view_path.exists():
                        available_views[view] = view_filename

            # 如果有足够的视图文件，添加到标注中
            if len(available_views) >= len(views):
                annotations[model_name] = {
                    "views": available_views,
                    "metadata": {
                    }
                }

        # 保存标注文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        print(f"Created annotation template for {len(annotations)} models at {output_file}")

    @staticmethod
    def validate_dataset(data_dir: str, annotation_file: Optional[str] = None) -> Dict[str, int]:
        """
        验证数据集完整性
        适配文件夹结构: model_name/front_view.png, model_name/side_view.png, model_name/top_view.png

        Returns:
            统计信息字典
        """
        stats = {
            "total_models": 0,
            "valid_models": 0,
            "missing_views": 0,
            "invalid_files": 0
        }

        data_dir = Path(data_dir)
        view_files = {
            "front": "front_view.png",
            "side": "side_view.png",
            "top": "top_view.png"
        }

        # 扫描所有模型文件夹
        model_folders = [d for d in data_dir.iterdir() if d.is_dir()]
        stats["total_models"] = len(model_folders)

        for model_folder in model_folders:
            model_name = model_folder.name
            valid_views = 0
            total_expected_views = len(view_files)

            for view, view_filename in view_files.items():
                image_path = model_folder / view_filename

                if image_path.exists():
                    try:
                        # 尝试打开图像验证有效性
                        Image.open(image_path)
                        valid_views += 1
                    except Exception as e:
                        stats["invalid_files"] += 1
                        logger.warning(f"Invalid image file {image_path}: {e}")
                else:
                    stats["missing_views"] += 1
                    logger.warning(f"Missing view file: {image_path}")

            if valid_views == total_expected_views:
                stats["valid_models"] += 1

        return stats


def create_sample_annotation():
    """创建示例标注文件，适配新的文件夹结构"""
    sample_annotation = {
        "demo_1": {
            "views": {
                "front": "front_view.png",
                "side": "side_view.png",
                "top": "top_view.png"
            },
            "metadata": {

            }
        },
        "demo_2": {
            "views": {
                "front": "front_view.png",
                "side": "side_view.png",
                "top": "top_view.png"
            },
            "metadata": {
            }
        }
    }

    return sample_annotation