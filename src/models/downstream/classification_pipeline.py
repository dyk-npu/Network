import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.models.encoders.image_encoder import create_image_encoder
from src.models.encoders.brep_encoder import create_brep_encoder
from src.models.fusion.multimodal_fusion import create_bimodal_fuser
from src.models.downstream.classification import create_classification_model, create_classification_loss, ClassificationMetrics


class MultiModalClassificationPipeline(nn.Module):
    """完整的多模态分类管道"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 编码器
        self.image_encoder = create_image_encoder(config)
        self.brep_encoder = create_brep_encoder(config)

        # 融合器
        self.fuser = create_bimodal_fuser(config)

        # 分类器
        self.classifier = create_classification_model(config)

        # 损失函数
        self.criterion = create_classification_loss(config)

    def forward(self, images: torch.Tensor, graphs: 'dgl.DGLGraph', labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        端到端前向传播

        Args:
            images: 多视图图像 [batch_size, num_views, channels, height, width]
            graphs: 批量DGL图
            labels: 可选的真实标签 [batch_size]

        Returns:
            包含分类结果和损失的字典
        """
        # 特征编码
        image_features = self.image_encoder(images)  # [batch_size, image_dim]
        brep_output = self.brep_encoder(graphs)
        brep_features = brep_output['features']      # [batch_size, brep_dim]

        # 特征融合
        fusion_output = self.fuser(image_features, brep_features)

        # 分类
        classification_output = self.classifier(fusion_output)

        results = {
            'logits': classification_output['logits'],
            'probabilities': classification_output['probabilities'],
            'predictions': torch.argmax(classification_output['logits'], dim=-1),
            'feature_importance': classification_output['feature_importance'],
            'image_logits': classification_output['image_logits'],
            'brep_logits': classification_output['brep_logits'],
            'fused_features': fusion_output['fused_features'],
            'image_features': image_features,
            'brep_features': brep_features
        }

        # 计算损失（训练时）
        if labels is not None:
            losses = self.criterion(classification_output, labels)
            results.update(losses)

        return results

    def predict(self, images: torch.Tensor, graphs: 'dgl.DGLGraph') -> Dict[str, torch.Tensor]:
        """推理模式"""
        self.eval()
        with torch.no_grad():
            results = self.forward(images, graphs)
            confidence = torch.max(results['probabilities'], dim=-1)[0]

            return {
                'predictions': results['predictions'],
                'probabilities': results['probabilities'],
                'confidence': confidence,
                'feature_importance': results['feature_importance']
            }

    def extract_features(self, images: torch.Tensor, graphs: 'dgl.DGLGraph') -> Dict[str, torch.Tensor]:
        """提取各阶段特征（用于分析）"""
        self.eval()
        with torch.no_grad():
            results = self.forward(images, graphs)

            return {
                'image_features': results['image_features'],
                'brep_features': results['brep_features'],
                'fused_features': results['fused_features'],
                'feature_importance': results['feature_importance']
            }


class ClassificationTrainer:
    """分类任务的训练器"""

    def __init__(self, model: MultiModalClassificationPipeline, model_config: Optional[ModelConfig] = None,
                 base_config: Optional[BaseConfig] = None):
        self.model = model
        if model_config is None:
            model_config = ModelConfig()
        if base_config is None:
            base_config = BaseConfig()

        self.model_config = model_config
        self.base_config = base_config

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_config.learning_rate,
            weight_decay=getattr(base_config, 'weight_decay', 1e-4)
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=getattr(base_config, 'num_epochs', 100),
            eta_min=base_config.learning_rate * 0.01
        )

        # 评估指标
        self.metrics = ClassificationMetrics(model_config.num_classes)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # 数据移到设备
            images = batch['images'].to(self.device)
            graphs = batch['brep_graph'].to(self.device)  # 使用正确的键名
            labels = batch['labels'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, graphs, labels)

            # 反向传播
            loss = outputs['total_loss']
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_main_loss += outputs['main_loss'].item()
            total_aux_loss += (outputs['image_loss'].item() + outputs['brep_loss'].item()) / 2

            # 更新指标
            predictions = outputs['predictions']
            self.metrics.update(predictions, labels)

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 更新学习率
        self.scheduler.step()

        # 计算平均损失和指标
        avg_loss = total_loss / len(dataloader)
        avg_main_loss = total_main_loss / len(dataloader)
        avg_aux_loss = total_aux_loss / len(dataloader)

        metrics = self.metrics.compute()

        return {
            'loss': avg_loss,
            'main_loss': avg_main_loss,
            'aux_loss': avg_aux_loss,
            'accuracy': metrics['accuracy'],
            'mean_accuracy': metrics['mean_accuracy'],
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def validate(self, dataloader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                graphs = batch['brep_graph'].to(self.device)  # 使用正确的键名
                labels = batch['labels'].to(self.device)

                outputs = self.model(images, graphs, labels)
                loss = outputs['total_loss']

                total_loss += loss.item()
                predictions = outputs['predictions']
                self.metrics.update(predictions, labels)

        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics.compute()

        return {
            'loss': avg_loss,
            'accuracy': metrics['accuracy'],
            'mean_accuracy': metrics['mean_accuracy'],
            'class_accuracies': metrics['class_accuracies']
        }

    def save_checkpoint(self, filepath: str, epoch: int, best_acc: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': best_acc,
            'config': self.config
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_accuracy']


def create_classification_pipeline(config: Optional[ModelConfig] = None) -> MultiModalClassificationPipeline:
    """创建分类管道的工厂函数"""
    return MultiModalClassificationPipeline(config)


if __name__ == "__main__":
    # 测试分类管道
    import dgl

    config = ModelConfig()
    pipeline = create_classification_pipeline(config)

    # 模拟数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 3, 224, 224)

    # 创建图数据
    graphs = []
    for _ in range(batch_size):
        num_nodes = 6
        src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
        dst = torch.tensor([1, 2, 3, 4, 5, 0, 2, 3, 4])
        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        graph = dgl.add_self_loop(graph)
        graph.ndata['x'] = torch.randn(num_nodes, 7)
        graphs.append(graph)

    batched_graphs = dgl.batch(graphs)
    labels = torch.randint(0, config.num_classes, (batch_size,))

    # 测试端到端前向传播
    with torch.no_grad():
        results = pipeline(images, batched_graphs, labels)

        print("分类管道测试结果:")
        print(f"  预测结果: {results['predictions']}")
        print(f"  真实标签: {labels}")
        print(f"  总损失: {results['total_loss'].item():.4f}")
        print(f"  主损失: {results['main_loss'].item():.4f}")
        print(f"  准确率: {(results['predictions'] == labels).float().mean().item():.4f}")

        # 测试推理
        pred_results = pipeline.predict(images, batched_graphs)
        print(f"  推理预测: {pred_results['predictions']}")
        print(f"  置信度: {pred_results['confidence']}")

    print("✓ 分类管道测试通过！")