import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np
import dgl

from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.models.encoders.image_encoder import create_image_encoder
from src.models.encoders.brep_encoder import BrepEncoder
from src.models.fusion.multimodal_fusion import create_bimodal_fuser
from src.models.downstream.retrieval import (
    create_retrieval_model,
    create_retrieval_loss,
    RetrievalMetrics
)


class MultiModalRetrievalPipeline(nn.Module):
    """多模态检索完整流水线"""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config

        # 编码器
        self.image_encoder = create_image_encoder(model_config)
        self.brep_encoder = BrepEncoder(model_config)

        # 融合模块
        self.fusion_module = create_bimodal_fuser(model_config)

        # 检索模块
        # 使用配置中的设置决定是否使用高级检索模型
        use_advanced = getattr(model_config, 'use_advanced_retrieval', True)
        self.retrieval_module = create_retrieval_model(model_config, use_advanced=use_advanced)

        # 损失函数
        temperature = getattr(model_config, 'contrastive_temperature', 0.07)
        self.criterion = create_retrieval_loss(temperature=temperature, use_advanced=use_advanced)

    def encode_features(self, images: torch.Tensor,
                       brep_graphs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码输入特征"""
        # 图像编码
        image_features = self.image_encoder(images)

        # Brep编码
        brep_output = self.brep_encoder(brep_graphs)
        brep_features = brep_output['features'] if isinstance(brep_output, dict) else brep_output

        # 多模态融合
        fusion_output = self.fusion_module(image_features, brep_features)

        return fusion_output

    def forward(self, query_images: torch.Tensor, query_brep: torch.Tensor,
                candidate_images: torch.Tensor, candidate_brep: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            query_images: 查询图像
            query_brep: 查询Brep
            candidate_images: 候选图像
            candidate_brep: 候选Brep
            labels: 正样本标签

        Returns:
            输出字典
        """
        # 编码查询特征
        query_features = self.encode_features(query_images, query_brep)

        # 编码候选特征
        candidate_features = self.encode_features(candidate_images, candidate_brep)

        # 检索
        retrieval_output = self.retrieval_module(query_features, candidate_features, labels)

        return retrieval_output

    def retrieve(self, query_images: torch.Tensor, query_brep: torch.Tensor,
                candidate_images: torch.Tensor, candidate_brep: torch.Tensor,
                k: int = 10, modality: str = 'fused') -> Dict[str, torch.Tensor]:
        """
        执行检索推理

        Args:
            query_images: 查询图像
            query_brep: 查询Brep
            candidate_images: 候选图像
            candidate_brep: 候选Brep
            k: 返回top-k结果
            modality: 检索模态

        Returns:
            检索结果
        """
        with torch.no_grad():
            # 编码特征
            query_features = self.encode_features(query_images, query_brep)
            candidate_features = self.encode_features(candidate_images, candidate_brep)

            # 执行检索
            results = self.retrieval_module.retrieve(
                query_features, candidate_features, k, modality
            )

            return results


class RetrievalTrainer:
    """检索模型训练器"""

    def __init__(self, model: MultiModalRetrievalPipeline,
                 model_config: ModelConfig, base_config: BaseConfig):
        self.model = model
        self.model_config = model_config
        self.base_config = base_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 评估指标
        self.metrics = RetrievalMetrics()

        # 日志
        self.logger = logging.getLogger(__name__)

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = getattr(self.base_config, 'optimizer', 'adamw')

        if optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.base_config.learning_rate,
                weight_decay=self.base_config.weight_decay
            )
        elif optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.base_config.learning_rate,
                weight_decay=self.base_config.weight_decay
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.base_config.learning_rate,
                momentum=0.9,
                weight_decay=self.base_config.weight_decay
            )

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_name = getattr(self.base_config, 'scheduler', 'cosine')

        if scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.base_config.num_epochs
            )
        elif scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.base_config.num_epochs // 3,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="训练")

        for batch in progress_bar:
            # 数据移动到设备
            images = batch['images'].to(self.device)
            brep_graphs = batch['brep_graph'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 构造候选集（在训练时，使用同一批次的其他样本作为候选）
            batch_size = images.size(0)

            # 为每个查询创建候选集（包含正样本和负样本）
            candidate_images = images  # 使用整个批次作为候选集
            candidate_brep = brep_graphs
            candidate_labels = labels

            # 创建正样本标签（每个查询的正样本是它自己在候选集中的位置）
            positive_labels = torch.arange(batch_size, device=self.device)

            # 前向传播
            output = self.model(
                images, brep_graphs,
                candidate_images, candidate_brep,
                positive_labels
            )

            # 计算损失
            loss_output = self.model.criterion(output, positive_labels)
            loss = loss_output['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({'Loss': loss.item()})

        # 学习率调度
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # 对于ReduceLROnPlateau，需要传入指标
            pass  # 在验证时调用
        else:
            self.scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.metrics.reset()

        # 清理之前的候选池释放内存
        if hasattr(self, '_candidate_pool'):
            del self._candidate_pool
        torch.cuda.empty_cache()  # 清理GPU缓存

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="验证")

            for batch in progress_bar:
                # 数据移动到设备
                images = batch['images'].to(self.device)
                brep_graphs = batch['brep_graph'].to(self.device)
                labels = batch['labels'].to(self.device)

                batch_size = images.size(0)

                # 构造候选集进行验证（限制大小避免OOM）
                max_candidate_batches = 20  # 减少候选池大小
                if not hasattr(self, '_candidate_pool') or len(self._candidate_pool['images']) < max_candidate_batches:
                    if not hasattr(self, '_candidate_pool'):
                        self._candidate_pool = {
                            'images': [],
                            'brep_graphs': [],
                            'labels': []
                        }

                    # 限制候选池大小，采用FIFO策略
                    if len(self._candidate_pool['images']) >= max_candidate_batches:
                        # 移除最旧的批次
                        self._candidate_pool['images'].pop(0)
                        self._candidate_pool['brep_graphs'].pop(0)
                        self._candidate_pool['labels'].pop(0)

                    self._candidate_pool['images'].append(images.cpu())  # 移到CPU节省GPU内存
                    self._candidate_pool['brep_graphs'].append(brep_graphs.cpu())
                    self._candidate_pool['labels'].append(labels.cpu())

                    # 如果候选池还不够大，跳过这个批次的评估
                    if len(self._candidate_pool['images']) < 3:  # 减少最小要求
                        continue

                # 使用候选池进行评估（限制数量避免OOM）
                max_candidates = min(5, len(self._candidate_pool['images']))  # 减少候选数量

                # 移回GPU进行计算
                candidate_images = torch.cat([img.to(self.device) for img in self._candidate_pool['images'][:max_candidates]], dim=0)

                # 优化图处理，减少内存占用
                candidate_graph_list = self._candidate_pool['brep_graphs'][:max_candidates]
                normalized_graphs = []
                for graph in candidate_graph_list:
                    # 移到GPU并清理特征
                    g = graph.to(self.device)
                    # 原地移除特征，避免克隆
                    if 'edge_agg' in g.ndata:
                        del g.ndata['edge_agg']
                    if 'encoded' in g.edata:
                        del g.edata['encoded']
                    normalized_graphs.append(g)

                candidate_brep = dgl.batch(normalized_graphs)
                candidate_labels = torch.cat([lbl.to(self.device) for lbl in self._candidate_pool['labels'][:max_candidates]], dim=0)

                # 执行检索
                retrieval_results = self.model.retrieve(
                    images, brep_graphs,
                    candidate_images, candidate_brep,
                    k=10, modality='fused'
                )

                # 更新评估指标
                self.metrics.update(
                    retrieval_results['indices'],
                    candidate_labels,
                    labels
                )

                num_batches += 1

        # 计算最终指标
        final_metrics = self.metrics.compute()

        # 验证结束后清理内存
        if hasattr(self, '_candidate_pool'):
            del self._candidate_pool
        torch.cuda.empty_cache()

        # 学习率调度（如果使用ReduceLROnPlateau）
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            metric_for_scheduler = final_metrics.get('mAP', 0.0)
            self.scheduler.step(metric_for_scheduler)

        return final_metrics

    def save_checkpoint(self, filepath: str, epoch: int,
                       best_metric: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': best_metric,
            'model_config': self.model_config,
            'base_config': self.base_config
        }

        torch.save(checkpoint, filepath)

        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_filepath)

    def load_checkpoint(self, filepath: str) -> Tuple[int, float]:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 获取当前模型的状态字典
        current_state = self.model.state_dict()
        checkpoint_state = checkpoint['model_state_dict']

        # 过滤出匹配的参数
        filtered_state = {}
        missing_keys = []
        unexpected_keys = []

        for key, param in checkpoint_state.items():
            if key in current_state:
                if current_state[key].shape == param.shape:
                    filtered_state[key] = param
                else:
                    self.logger.warning(f"跳过形状不匹配的参数: {key} "
                                      f"(检查点: {param.shape}, 当前: {current_state[key].shape})")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)

        # 检查缺失的键
        for key in current_state:
            if key not in filtered_state:
                missing_keys.append(key)

        # 检查加载比例，如果太少则警告
        load_ratio = len(filtered_state) / len(current_state) if len(current_state) > 0 else 0

        if load_ratio < 0.5:
            self.logger.warning(f"检查点兼容性较低 ({load_ratio:.1%})，可能影响模型性能")
            self.logger.warning("建议删除或更新检查点文件，或从头开始训练")

        # 加载过滤后的状态字典
        self.model.load_state_dict(filtered_state, strict=False)

        # 打印加载信息
        if missing_keys:
            self.logger.warning(f"缺失的参数: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            self.logger.warning(f"意外的参数: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        self.logger.info(f"成功加载 {len(filtered_state)}/{len(current_state)} 个参数 ({load_ratio:.1%})")

        # 尝试加载优化器和调度器状态（如果存在且兼容）
        try:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            self.logger.warning(f"无法加载优化器状态: {e}")

        try:
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            self.logger.warning(f"无法加载调度器状态: {e}")

        return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)

    def load_classification_checkpoint(self, filepath: str) -> None:
        """从分类模型检查点加载兼容的组件"""
        self.logger.info(f"从分类检查点加载兼容组件: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        classification_state = checkpoint['model_state_dict']

        # 获取当前检索模型的状态字典
        retrieval_state = self.model.state_dict()

        # 映射分类模型组件到检索模型
        component_mapping = {
            # 图像编码器
            'image_encoder': 'image_encoder',
            # Brep编码器
            'brep_encoder': 'brep_encoder',
            # 融合模块：分类模型中叫 fuser，检索模型中叫 fusion_module
            'fuser': 'fusion_module'
        }

        # 加载兼容的组件
        loaded_components = []
        for class_component, retrieval_component in component_mapping.items():
            component_loaded = False
            for key, value in classification_state.items():
                if key.startswith(f"{class_component}."):
                    # 将分类模型的键映射到检索模型的键
                    new_key = key.replace(f"{class_component}.", f"{retrieval_component}.")
                    if new_key in retrieval_state:
                        retrieval_state[new_key] = value
                        component_loaded = True

            if component_loaded:
                loaded_components.append(retrieval_component)

        # 加载更新后的状态字典
        self.model.load_state_dict(retrieval_state, strict=False)

        self.logger.info(f"成功加载组件: {', '.join(loaded_components)}")
        self.logger.info("检索专用组件将从随机初始化开始训练")


def create_retrieval_pipeline(model_config: ModelConfig) -> MultiModalRetrievalPipeline:
    """创建检索流水线的工厂函数"""
    return MultiModalRetrievalPipeline(model_config)


if __name__ == "__main__":
    # 测试检索流水线
    import dgl

    print("🧪 测试检索流水线...")

    model_config = ModelConfig()
    base_config = BaseConfig()

    # 创建流水线
    pipeline = create_retrieval_pipeline(model_config)

    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.to(device)

    print(f"  使用设备: {device}")

    # 创建模拟DGL图
    def create_mock_graph(num_nodes=8):
        src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])
        dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 2, 3, 4])
        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        graph = dgl.add_self_loop(graph)
        graph.ndata['x'] = torch.randn(num_nodes, 7)
        return graph

    # 模拟数据
    batch_size = 4  # 减小批次大小
    num_views = 3
    height, width = 224, 224

    # 创建图像数据
    images = torch.randn(batch_size, num_views, 3, height, width).to(device)

    # 创建DGL图数据
    graphs = []
    for _ in range(batch_size):
        graphs.append(create_mock_graph())
    brep_graphs = dgl.batch(graphs).to(device)

    print(f"  图像数据形状: {images.shape}")
    print(f"  图数据: {brep_graphs.batch_size} 个图")

    # 测试特征编码
    with torch.no_grad():
        print("  1. 测试特征编码...")

        # 创建查询图（前2个）
        query_graphs = []
        for i in range(2):
            query_graphs.append(create_mock_graph())
        query_brep_graphs = dgl.batch(query_graphs).to(device)

        query_features = pipeline.encode_features(images[:2], query_brep_graphs)
        candidate_features = pipeline.encode_features(images, brep_graphs)

        print(f"     查询特征形状: {query_features['fused_features'].shape}")
        print(f"     候选特征形状: {candidate_features['fused_features'].shape}")

        # 测试检索功能
        print("  2. 测试检索功能...")
        retrieval_results = pipeline.retrieve(
            images[:2], query_brep_graphs,
            images, brep_graphs,
            k=3, modality='fused'
        )

        print(f"     检索结果形状: {retrieval_results['indices'].shape}")
        print(f"     检索相似度形状: {retrieval_results['similarities'].shape}")
        print(f"     Top-3索引: {retrieval_results['indices'][0].tolist()}")

    print("✓ 检索流水线测试通过！")