import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
import dgl

logger = logging.getLogger(__name__)

# 导入统一的类别定义和验证函数
from config.model_config import (
    CAD_PART_CLASSES,
    get_category_mapping,
    get_index_to_category,
    validate_category,
    validate_label,
    category_to_label,
    label_to_category
)


class MultiModalCADDataset(Dataset):
    """
    统一的多模态CAD零件数据集
    支持加载图像（多视图）、B-rep（预处理的DGL图）和文本描述等多种模态数据

    数据结构:
    - 图像: model_name/front_view.png, model_name/side_view.png, model_name/top_view.png
    - B-rep: model_name.bin (DGL图格式)
    - 文本: annotations.json (包含描述信息)
    """

    def __init__(
        self,
        data_dir: str = None,
        brep_dir: Optional[str] = None,
        annotation_file: Optional[str] = None,
        modalities: List[str] = ["image", "brep"],
        transform=None,
        views: List[str] = ["front", "side", "top"],
        view_files: Dict[str, str] = None
    ):
        """
        Args:
            data_dir: 图像数据根目录，包含以模型名称命名的文件夹
            brep_dir: B-rep数据目录，包含预处理后的.bin文件
            annotation_file: 标注文件路径（JSON格式），可选
            modalities: 要加载的模态列表 ["image", "brep", "text"]
            transform: 图像变换函数
            views: 视图名称列表
            view_files: 视图文件名映射 {view_name: filename}
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.brep_dir = Path(brep_dir) if brep_dir else None
        self.modalities = modalities
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

        # 验证模态参数（如果没有标注文件，则需要指定目录）
        if not self.annotations:
            if "image" in self.modalities and not self.data_dir:
                raise ValueError("data_dir is required when 'image' is in modalities and no annotation file is provided")
            if "brep" in self.modalities and not self.brep_dir:
                raise ValueError("brep_dir is required when 'brep' is in modalities and no annotation file is provided")
        self.samples = self._prepare_samples()

        logger.info(f"Loaded {len(self.samples)} samples with modalities: {self.modalities}")

    def _load_annotations(self, annotation_file: str) -> Dict:
        """加载标注文件"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        return annotations

    def _prepare_samples(self) -> List[Dict]:
        """准备样本列表，支持多种模态数据"""
        samples = []

        # 如果有标注文件，直接从标注文件中加载样本
        if self.annotations:
            return self._prepare_samples_from_annotations()

        # 否则使用传统的目录扫描方式
        return self._prepare_samples_from_directories()

    def _prepare_samples_from_annotations(self) -> List[Dict]:
        """从标注文件准备样本列表"""
        samples = []

        # 遍历标注文件中的所有模型（跳过_categories_info）
        for item_id, annotation_data in self.annotations.items():
            if item_id.startswith("_"):  # 跳过元信息
                continue

            sample = {
                "item_id": item_id,
                "metadata": annotation_data.get("metadata", {}),
                "available_modalities": []
            }

            available_data = annotation_data.get("available_data", {})

            # 检查图像模态
            if "image" in self.modalities:
                views_data = available_data.get("views", {})
                if views_data:
                    views = {}
                    valid_views = 0
                    for view in self.views:
                        view_key = view  # front, side, top
                        if view_key in views_data:
                            image_path = Path(views_data[view_key])
                            if image_path.exists():
                                views[view] = image_path
                                valid_views += 1
                            else:
                                logger.warning(f"Image file not found: {image_path}")

                    if valid_views >= len(self.views):
                        sample["views"] = views
                        sample["available_modalities"].append("image")
                    else:
                        logger.warning(f"Model {item_id}: only {valid_views}/{len(self.views)} views available")

            # 检查B-rep模态
            if "brep" in self.modalities:
                brep_file = available_data.get("brep")
                if brep_file:
                    brep_path = Path(brep_file)
                    if brep_path.exists():
                        sample["brep_path"] = brep_path
                        sample["available_modalities"].append("brep")
                    else:
                        logger.warning(f"Brep file not found: {brep_path}")

            # 检查文本模态
            if "text" in self.modalities:
                if sample["metadata"]:  # 如果有metadata就认为有文本
                    sample["available_modalities"].append("text")

            # 只保留至少有一种指定模态的样本
            required_modalities = set(self.modalities)
            available_modalities = set(sample["available_modalities"])
            if required_modalities.intersection(available_modalities):
                samples.append(sample)
            else:
                logger.warning(f"Skipping model {item_id}: no required modalities available")

        return samples

    def _prepare_samples_from_directories(self) -> List[Dict]:
        """从目录扫描准备样本列表（传统方式）"""
        samples = []
        model_names = set()

        # 收集所有可用的模型名称
        if "image" in self.modalities and self.data_dir:
            image_models = {d.name for d in self.data_dir.iterdir() if d.is_dir()}
            model_names.update(image_models)

        if "brep" in self.modalities and self.brep_dir:
            brep_models = {f.stem for f in self.brep_dir.glob("*.bin")}
            model_names.update(brep_models)

        # 为每个模型准备样本
        for model_name in model_names:
            sample = {
                "item_id": model_name,
                "metadata": self.annotations.get(model_name, {}).get("metadata", {}),
                "available_modalities": []
            }

            # 检查图像模态
            if "image" in self.modalities and self.data_dir:
                model_folder = self.data_dir / model_name
                if model_folder.exists() and model_folder.is_dir():
                    views = {}
                    valid_views = 0
                    for view in self.views:
                        view_filename = self.view_files.get(view)
                        if view_filename:
                            image_path = model_folder / view_filename
                            if image_path.exists():
                                views[view] = image_path
                                valid_views += 1

                    if valid_views >= len(self.views):
                        sample["views"] = views
                        sample["available_modalities"].append("image")
                    else:
                        logger.warning(f"Model {model_name}: only {valid_views}/{len(self.views)} views available")

            # 检查B-rep模态
            if "brep" in self.modalities and self.brep_dir:
                brep_path = self.brep_dir / f"{model_name}.bin"
                if brep_path.exists():
                    sample["brep_path"] = brep_path
                    sample["available_modalities"].append("brep")

            # 检查文本模态
            if "text" in self.modalities:
                if model_name in self.annotations:
                    sample["available_modalities"].append("text")

            # 只保留至少有一种指定模态的样本
            required_modalities = set(self.modalities)
            available_modalities = set(sample["available_modalities"])
            if required_modalities.intersection(available_modalities):
                samples.append(sample)
            else:
                logger.warning(f"Skipping model {model_name}: no required modalities available")

        return samples

    def get_type_statistics(self) -> Dict[str, int]:
        """获取数据集中类型标签的统计信息"""
        type_counts = {}

        for sample in self.samples:
            sample_annotation = self.annotations.get(sample["item_id"], {})
            metadata = sample_annotation.get("metadata", {})

            # 统计主要分类
            category = metadata.get("category", "unknown")
            type_counts[category] = type_counts.get(category, 0) + 1

        return {
            "categories": type_counts,
            "total_samples": len(self.samples)
        }

    @staticmethod
    def get_classes_from_directory(data_dir: str) -> List[str]:
        """从数据目录中自动获取类别名称（文件夹名称）"""
        data_path = Path(data_dir)
        if data_path.exists():
            categories = [d.name for d in data_path.iterdir() if d.is_dir()]
            return sorted(categories)  # 排序确保一致性
        return CAD_PART_CLASSES

    @staticmethod
    def get_category_mapping() -> Dict[str, int]:
        """获取类别名称到索引的映射"""
        return get_category_mapping()

    @staticmethod
    def get_index_to_category() -> Dict[int, str]:
        """获取索引到类别名称的映射"""
        return get_index_to_category()

    @staticmethod
    def category_to_label(category: str) -> int:
        """将类别字符串转换为数字标签"""
        try:
            return category_to_label(category)
        except ValueError as e:
            logger.warning(f"Category validation failed: {e}")
            return -1  # 返回-1表示未知类别

    @staticmethod
    def label_to_category(label: int) -> str:
        """将数字标签转换为类别字符串"""
        try:
            return label_to_category(label)
        except ValueError as e:
            logger.warning(f"Label validation failed: {e}")
            return "unknown"

    def validate_categories(self) -> Dict[str, List[str]]:
        """验证数据集中的类别是否都在预定义类别中"""
        valid_categories = []
        invalid_categories = []

        for sample in self.samples:
            annotation_data = self.annotations.get(sample["item_id"], {})
            metadata = annotation_data.get("metadata", {})
            category = metadata.get("category", "")

            if validate_category(category):
                if category not in valid_categories:
                    valid_categories.append(category)
            else:
                if category not in invalid_categories:
                    invalid_categories.append(category)

        return {
            "valid_categories": valid_categories,
            "invalid_categories": invalid_categories,
            "predefined_classes": CAD_PART_CLASSES
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        获取多模态数据样本

        Returns:
            Dict包含:
            - images: 多视图图像张量 [num_views, C, H, W] (如果包含image模态)
            - brep_graph: DGL图对象 (如果包含brep模态)
            - text: 文本描述 (如果包含text模态)
            - type: 零件类型/分类标签 (字符串)
            - label: 数字标签索引 (用于训练)
            - item_id: 零件ID
            - metadata: 元数据
            - available_modalities: 可用的模态列表
        """
        sample = self.samples[idx]
        result = {
            "item_id": sample["item_id"],
            "metadata": sample["metadata"],
            "available_modalities": sample["available_modalities"]
        }

        # 加载图像模态
        if "image" in self.modalities and "image" in sample["available_modalities"]:
            images = []
            for view in self.views:
                if view in sample["views"]:
                    image_path = sample["views"][view]
                    image = Image.open(image_path).convert("RGB")
                    # Resize图像到224x224 (ViT标准输入尺寸)
                    image = image.resize((224, 224), Image.LANCZOS)
                    images.append(image)
                else:
                    # 如果缺少某个视图，创建空白图像
                    images.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))

            # 应用变换
            if self.transform:
                images = self.transform(images)
            else:
                # 默认变换：转换为张量并标准化到[0,1]
                images = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in images])

            result["images"] = images

        # 加载B-rep模态
        if "brep" in self.modalities and "brep" in sample["available_modalities"]:
            try:
                graphs, _ = dgl.data.utils.load_graphs(str(sample["brep_path"]))
                graph = graphs[0]  # 取第一个图

                # 确保图的特征是Float类型，避免与模型权重类型不匹配
                if 'x' in graph.ndata:
                    graph.ndata['x'] = graph.ndata['x'].float()
                if 'x' in graph.edata:
                    graph.edata['x'] = graph.edata['x'].float()

                result["brep_graph"] = graph
            except Exception as e:
                logger.warning(f"Failed to load brep graph for {sample['item_id']}: {e}")
                result["brep_graph"] = None

        # 加载文本模态
        if "text" in self.modalities and "text" in sample["available_modalities"]:
            text_data = self.annotations.get(sample["item_id"], {})
            result["text"] = text_data.get("description", "")
            result["labels"] = text_data.get("labels", {})

        # 加载类型标签（无论是否包含text模态都可能需要）
        annotation_data = self.annotations.get(sample["item_id"], {})
        metadata = annotation_data.get("metadata", {})
        category_str = metadata.get("category", "")

        # 转换字符串类别为数字标签
        result["type"] = category_str
        result["label"] = self.category_to_label(category_str)

        return result


class MultiModalCADDataLoader:
    """多模态CAD数据加载器工具类"""

    @staticmethod
    def create_annotation_template(
        image_dir: str = None,
        brep_dir: str = None,
        output_file: str = "annotations.json",
        views: List[str] = ["front", "side", "top"],
        view_files: Dict[str, str] = None
    ):
        """
        创建多模态标注文件模板

        Args:
            image_dir: 图像目录，包含以模型名称命名的子文件夹
            brep_dir: B-rep数据目录，包含.bin文件
            output_file: 输出标注文件路径
            views: 视图列表
            view_files: 视图文件名映射
        """
        annotations = {}
        model_names = set()

        # 默认视图文件映射
        if view_files is None:
            view_files = {
                "front": "front_view.png",
                "side": "side_view.png",
                "top": "top_view.png"
            }

        # 收集所有模型名称
        if image_dir:
            image_dir = Path(image_dir)
            model_folders = [d for d in image_dir.iterdir() if d.is_dir()]
            model_names.update([d.name for d in model_folders])

        if brep_dir:
            brep_dir = Path(brep_dir)
            brep_files = list(brep_dir.glob("*.bin"))
            model_names.update([f.stem for f in brep_files])

        # 为每个模型创建标注
        for model_name in model_names:
            annotation = {
                "metadata": {
                    "description": f"CAD part: {model_name}",
                    "category": ""  # 主要分类（如：connector, bracket, gear等）
                },
                "available_data": {}
            }

            # 检查图像数据
            if image_dir:
                model_folder = image_dir / model_name
                if model_folder.exists() and model_folder.is_dir():
                    available_views = {}
                    for view in views:
                        view_filename = view_files.get(view)
                        if view_filename:
                            view_path = model_folder / view_filename
                            if view_path.exists():
                                available_views[view] = view_filename

                    if available_views:
                        annotation["available_data"]["views"] = available_views

            # 检查B-rep数据
            if brep_dir:
                brep_path = brep_dir / f"{model_name}.bin"
                if brep_path.exists():
                    annotation["available_data"]["brep"] = f"{model_name}.bin"

            annotations[model_name] = annotation

        # 保存标注文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        print(f"Created multi-modal annotation template for {len(annotations)} models at {output_file}")

    @staticmethod
    def validate_dataset(
        data_dir: str = None,
        brep_dir: str = None,
        annotation_file: Optional[str] = None,
        modalities: List[str] = ["image", "brep"]
    ) -> Dict[str, int]:
        """
        验证多模态数据集完整性

        Returns:
            统计信息字典
        """
        stats = {
            "total_models": 0,
            "valid_models": 0,
            "missing_views": 0,
            "invalid_files": 0,
            "modality_stats": {}
        }

        model_names = set()

        # 收集所有模型名称
        if "image" in modalities and data_dir:
            data_dir = Path(data_dir)
            model_folders = [d for d in data_dir.iterdir() if d.is_dir()]
            model_names.update([d.name for d in model_folders])

        if "brep" in modalities and brep_dir:
            brep_dir = Path(brep_dir)
            brep_files = list(brep_dir.glob("*.bin"))
            model_names.update([f.stem for f in brep_files])

        stats["total_models"] = len(model_names)

        view_files = {
            "front": "front_view.png",
            "side": "side_view.png",
            "top": "top_view.png"
        }

        # 初始化模态统计
        for modality in modalities:
            stats["modality_stats"][modality] = {"available": 0, "valid": 0}

        for model_name in model_names:
            model_valid = True
            available_modalities = []

            # 验证图像模态
            if "image" in modalities and data_dir:
                model_folder = data_dir / model_name
                if model_folder.exists() and model_folder.is_dir():
                    valid_views = 0
                    total_expected_views = len(view_files)

                    for view, view_filename in view_files.items():
                        image_path = model_folder / view_filename
                        if image_path.exists():
                            try:
                                Image.open(image_path)
                                valid_views += 1
                            except Exception as e:
                                stats["invalid_files"] += 1
                                logger.warning(f"Invalid image file {image_path}: {e}")
                        else:
                            stats["missing_views"] += 1

                    if valid_views == total_expected_views:
                        stats["modality_stats"]["image"]["valid"] += 1
                        available_modalities.append("image")
                    if valid_views > 0:
                        stats["modality_stats"]["image"]["available"] += 1

            # 验证B-rep模态
            if "brep" in modalities and brep_dir:
                brep_path = brep_dir / f"{model_name}.bin"
                if brep_path.exists():
                    try:
                        graphs, _ = dgl.data.utils.load_graphs(str(brep_path))
                        if graphs:
                            stats["modality_stats"]["brep"]["valid"] += 1
                            available_modalities.append("brep")
                        stats["modality_stats"]["brep"]["available"] += 1
                    except Exception as e:
                        stats["invalid_files"] += 1
                        logger.warning(f"Invalid brep file {brep_path}: {e}")

            # 如果所有要求的模态都可用，认为是有效样本
            if set(modalities).issubset(set(available_modalities)):
                stats["valid_models"] += 1

        return stats


def create_sample_annotation():
    """创建多模态示例标注文件"""
    sample_annotation = {
        "demo_1": {
            "metadata": {
                "description": "Electrical connector",
                "category": "connector"
            },
            "available_data": {
                "views": {
                    "front": "front_view.png",
                    "side": "side_view.png",
                    "top": "top_view.png"
                },
                "brep": "demo_1.bin"
            }
        },
        "demo_2": {
            "metadata": {
                "description": "L-shaped mounting bracket",
                "category": "bracket"
            },
            "available_data": {
                "views": {
                    "front": "front_view.png",
                    "side": "side_view.png",
                    "top": "top_view.png"
                },
                "brep": "demo_2.bin"
            }
        }
    }

    return sample_annotation


def multimodal_collate_fn(batch):
    """
    多模态数据的批处理整理函数
    处理不同模态的数据并转换标签格式
    """
    batch_size = len(batch)
    collated = {}

    # 收集所有字段
    for key in batch[0].keys():
        if key == "images":
            # 处理图像数据
            images = []
            for item in batch:
                if "images" in item and item["images"] is not None:
                    images.append(item["images"])
            if images:
                collated["images"] = torch.stack(images)

        elif key == "brep_graph":
            # 处理DGL图数据
            graphs = []
            for item in batch:
                if "brep_graph" in item and item["brep_graph"] is not None:
                    graphs.append(item["brep_graph"])
                else:
                    # 如果某个样本没有B-rep图，我们需要处理这种情况
                    logger.warning(f"Missing brep_graph for item {item.get('item_id', 'unknown')}")

            if graphs:
                if len(graphs) != len(batch):
                    logger.warning(f"Batch size mismatch: {len(batch)} samples but only {len(graphs)} graphs")
                collated["brep_graph"] = dgl.batch(graphs)
            else:
                collated["brep_graph"] = None

        elif key == "label":
            # 处理标签 - 转换为张量用于训练
            labels = []
            for item in batch:
                label = item.get("label", -1)
                if label == -1:  # 处理未知类别
                    logger.warning(f"Unknown category for item {item.get('item_id', 'unknown')}")
                labels.append(label)
            collated["labels"] = torch.tensor(labels, dtype=torch.long)

        elif key in ["type", "text", "item_id", "metadata", "available_modalities"]:
            # 处理其他字段为列表
            collated[key] = [item.get(key, "") for item in batch]

    return collated