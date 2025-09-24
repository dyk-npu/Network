import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from config.model_config import ModelConfig


class MultiModalClassifier(nn.Module):
    """基于多模态融合特征的分类器"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 分类头网络
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.fusion_dim // 2),
            nn.Dropout(config.dropout),

            nn.Linear(config.fusion_dim // 2, config.fusion_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.fusion_dim // 4),
            nn.Dropout(config.dropout),

            nn.Linear(config.fusion_dim // 4, config.num_classes)
        )

        # 辅助分类器（使用单模态特征）
        self.image_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 4, config.num_classes)
        )

        self.brep_classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 4, config.num_classes)
        )

        # 特征重要性注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(config.fusion_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, fusion_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            fusion_output: 来自BiModalFuser的输出，包含：
                - fused_features: 融合特征 [batch_size, fusion_dim]
                - image_enhanced: 增强的图像特征 [batch_size, fusion_dim]
                - brep_enhanced: 增强的Brep特征 [batch_size, fusion_dim]

        Returns:
            Dict包含:
                - logits: 主分类器输出 [batch_size, num_classes]
                - image_logits: 图像分类器输出 [batch_size, num_classes]
                - brep_logits: Brep分类器输出 [batch_size, num_classes]
                - probabilities: 主分类器概率 [batch_size, num_classes]
                - feature_importance: 特征重要性权重 [batch_size, 1]
        """
        fused_features = fusion_output['fused_features']
        image_enhanced = fusion_output['image_enhanced']
        brep_enhanced = fusion_output['brep_enhanced']

        # 计算特征重要性权重
        feature_importance = self.feature_attention(fused_features)

        # 应用注意力权重
        attended_features = fused_features * feature_importance

        # 主分类器
        main_logits = self.classifier(attended_features)

        # 辅助分类器
        image_logits = self.image_classifier(image_enhanced)
        brep_logits = self.brep_classifier(brep_enhanced)

        # 计算概率
        main_probabilities = F.softmax(main_logits, dim=-1)

        return {
            'logits': main_logits,
            'image_logits': image_logits,
            'brep_logits': brep_logits,
            'probabilities': main_probabilities,
            'feature_importance': feature_importance
        }

    def predict(self, fusion_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """推理模式，返回预测结果"""
        with torch.no_grad():
            output = self.forward(fusion_output)

            # 获取预测类别
            predictions = torch.argmax(output['logits'], dim=-1)
            confidence = torch.max(output['probabilities'], dim=-1)[0]

            return {
                'predictions': predictions,
                'confidence': confidence,
                'probabilities': output['probabilities'],
                'feature_importance': output['feature_importance']
            }


class ClassificationLoss(nn.Module):
    """分类任务的组合损失函数"""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 主要损失：带标签平滑的交叉熵
        self.main_criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(config, 'label_smoothing', 0.1)
        )

        # 辅助损失：标准交叉熵
        self.aux_criterion = nn.CrossEntropyLoss()

        # 损失权重
        self.main_weight = 1.0
        self.image_weight = 0.3
        self.brep_weight = 0.3

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算组合损失

        Args:
            outputs: 分类器输出
            targets: 真实标签 [batch_size]

        Returns:
            损失字典
        """
        # 主损失
        main_loss = self.main_criterion(outputs['logits'], targets)

        # 辅助损失
        image_loss = self.aux_criterion(outputs['image_logits'], targets)
        brep_loss = self.aux_criterion(outputs['brep_logits'], targets)

        # 总损失
        total_loss = (
            self.main_weight * main_loss +
            self.image_weight * image_loss +
            self.brep_weight * brep_loss
        )

        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'image_loss': image_loss,
            'brep_loss': brep_loss
        }


class ClassificationMetrics:
    """分类任务的评估指标"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置统计"""
        self.correct = 0
        self.total = 0
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """更新统计"""
        self.predictions.append(predictions.cpu())
        self.targets.append(targets.cpu())

        # 整体准确率
        correct = (predictions == targets).sum().item()
        self.correct += correct
        self.total += targets.size(0)

        # 每类准确率
        for i in range(self.num_classes):
            class_mask = (targets == i)
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                self.class_correct[i] += (class_predictions == i).sum().item()
                self.class_total[i] += class_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        # 整体准确率
        accuracy = self.correct / self.total if self.total > 0 else 0.0

        # 每类准确率
        class_accuracies = []
        for i in range(self.num_classes):
            if self.class_total[i] > 0:
                class_acc = self.class_correct[i] / self.class_total[i]
                class_accuracies.append(class_acc.item())
            else:
                class_accuracies.append(0.0)

        # 平均准确率
        mean_accuracy = sum(class_accuracies) / len(class_accuracies)

        # 计算混淆矩阵相关指标
        all_predictions = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)

        # 计算F1分数等
        metrics = {
            'accuracy': accuracy,
            'mean_accuracy': mean_accuracy,
            'class_accuracies': class_accuracies
        }

        return metrics


def create_classification_model(config: Optional[ModelConfig] = None) -> MultiModalClassifier:
    """创建分类模型的工厂函数"""
    return MultiModalClassifier(config)


def create_classification_loss(config: Optional[ModelConfig] = None) -> ClassificationLoss:
    """创建分类损失函数的工厂函数"""
    return ClassificationLoss(config)


# 导入统一的类别定义
from config.model_config import CAD_PART_CLASSES


if __name__ == "__main__":
    # 测试分类模型
    config = ModelConfig()
    classifier = create_classification_model(config)
    criterion = create_classification_loss(config)

    # 模拟融合特征输入
    batch_size = 4
    fusion_output = {
        'fused_features': torch.randn(batch_size, config.fusion_dim),
        'image_enhanced': torch.randn(batch_size, config.fusion_dim),
        'brep_enhanced': torch.randn(batch_size, config.fusion_dim)
    }

    # 模拟标签
    targets = torch.randint(0, config.num_classes, (batch_size,))

    # 前向传播
    with torch.no_grad():
        outputs = classifier(fusion_output)
        losses = criterion(outputs, targets)

        print("分类模型测试结果:")
        print(f"  主分类器输出形状: {outputs['logits'].shape}")
        print(f"  概率输出形状: {outputs['probabilities'].shape}")
        print(f"  特征重要性形状: {outputs['feature_importance'].shape}")
        print(f"  总损失: {losses['total_loss'].item():.4f}")

        # 测试预测
        predictions = classifier.predict(fusion_output)
        print(f"  预测结果: {predictions['predictions']}")
        print(f"  置信度: {predictions['confidence']}")

    print("✓ 分类模型测试通过！")