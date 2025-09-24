#!/usr/bin/env python3
"""
CADNET多模态分类模型训练脚本
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.data.multimodal_dataset import MultiModalCADDataset, multimodal_collate_fn
from src.models.downstream.classification_pipeline import MultiModalClassificationPipeline, ClassificationTrainer


def setup_logging(log_dir: str, experiment_name: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{experiment_name}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(annotation_file: str, config: BaseConfig, train_ratio: float = 0.8):
    """创建数据加载器"""
    logger = logging.getLogger(__name__)

    # 创建数据集
    logger.info(f"加载数据集: {annotation_file}")
    dataset = MultiModalCADDataset(
        annotation_file=annotation_file,
        modalities=["image", "brep"],
        transform=None  # TODO: 添加数据增强
    )

    logger.info(f"数据集大小: {len(dataset)}")

    # 获取类别统计
    stats = dataset.get_type_statistics()
    logger.info(f"类别统计: {len(stats['categories'])} 个类别")
    for category, count in list(stats['categories'].items())[:10]:
        logger.info(f"  {category}: {count} 个样本")

    # 划分训练集和验证集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, dataset


def load_checkpoint(checkpoint_path: str, model, trainer, logger):
    """加载检查点"""
    logger.info(f"从检查点恢复训练: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器状态（如果存在）
    if 'scheduler_state_dict' in checkpoint and hasattr(trainer, 'scheduler'):
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 返回起始epoch和最佳指标
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_accuracy = checkpoint.get('best_accuracy', 0.0)

    logger.info(f"恢复训练从epoch {start_epoch}, 最佳准确率: {best_accuracy:.4f}")

    return start_epoch, best_accuracy


def train_model(config: BaseConfig, model_config: ModelConfig, train_loader, val_loader,
                experiment_name: str, logger):
    """训练模型"""

    # 创建模型
    logger.info("初始化模型...")
    model = MultiModalClassificationPipeline(model_config)

    # 创建训练器
    trainer = ClassificationTrainer(model, model_config, config)

    # 初始化训练参数
    start_epoch = 0
    best_accuracy = 0.0

    # 恢复训练（如果指定了检查点）
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        start_epoch, best_accuracy = load_checkpoint(
            config.resume_from_checkpoint, model, trainer, logger
        )

    # 创建tensorboard writer
    writer = SummaryWriter(os.path.join(config.tensorboard_log_dir, experiment_name))

    # 保存配置
    config_save_path = os.path.join(config.model_save_dir, experiment_name, "config.json")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

    with open(config_save_path, 'w') as f:
        json.dump({
            "model_config": model_config.__dict__,
            "base_config": config.__dict__
        }, f, indent=2)

    # 训练循环
    patience = 10
    patience_counter = 0

    logger.info("开始训练...")

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        # 训练一个epoch
        train_metrics = trainer.train_epoch(train_loader)

        # 记录训练指标
        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch)

        # 记录模型参数直方图和梯度（可配置）
        if config.log_histograms and (epoch + 1) % config.histogram_freq == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"Parameters/{name}", param.data, epoch)
                    if config.log_gradients:
                        writer.add_histogram(f"Gradients/{name}", param.grad.data, epoch)

        # 记录学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']
        writer.add_scalar("Train/learning_rate", current_lr, epoch)

        logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.4f}, "
                   f"LR: {current_lr:.6f}")

        # 验证
        if (epoch + 1) % config.eval_interval == 0:
            val_metrics = trainer.validate(val_loader)

            # 记录验证指标
            for key, value in val_metrics.items():
                writer.add_scalar(f"Val/{key}", value, epoch)

            # 记录训练和验证损失的比较
            if 'loss' in train_metrics and 'loss' in val_metrics:
                writer.add_scalars("Loss_Comparison", {
                    'Train': train_metrics['loss'],
                    'Validation': val_metrics['loss']
                }, epoch)

            # 记录训练和验证准确率的比较
            if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
                writer.add_scalars("Accuracy_Comparison", {
                    'Train': train_metrics['accuracy'],
                    'Validation': val_metrics['accuracy']
                }, epoch)

            logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}")

            # 保存最佳模型
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                patience_counter = 0

                model_save_path = os.path.join(
                    config.model_save_dir, experiment_name,
                    f"best_model_acc_{val_metrics['accuracy']:.4f}.pth"
                )

                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'config': model_config,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }

                # 保存调度器状态（如果存在）
                if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                    checkpoint_dict['scheduler_state_dict'] = trainer.scheduler.state_dict()

                torch.save(checkpoint_dict, model_save_path)

                logger.info(f"保存最佳模型: {model_save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"验证准确率连续{patience}个epoch没有提升，提前停止训练")
                break

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.model_save_dir, experiment_name,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )

            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'best_accuracy': best_accuracy,
                'config': model_config
            }

            # 添加验证指标（如果存在）
            if 'val_metrics' in locals():
                checkpoint_dict['val_metrics'] = val_metrics

            # 保存调度器状态（如果存在）
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = trainer.scheduler.state_dict()

            torch.save(checkpoint_dict, checkpoint_path)

            logger.info(f"保存检查点: {checkpoint_path}")

    writer.close()
    logger.info(f"训练完成! 最佳验证准确率: {best_accuracy:.4f}")

    return model, best_accuracy


def test_model(model_path: str, test_loader, logger):
    """测试模型"""
    logger.info(f"加载模型: {model_path}")

    # 加载模型
    checkpoint = torch.load(model_path)
    config = checkpoint['config']

    model = MultiModalClassificationPipeline(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 测试
    total_correct = 0
    total_samples = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            images = batch['images'].to(device)
            graphs = batch['brep_graph'].to(device)
            labels = batch['labels'].to(device)

            # 预测
            results = model.predict(images, graphs)
            predictions = results['predictions']

            # 统计
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # 每类统计
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0

                if pred.item() == label_item:
                    class_correct[label_item] += 1
                class_total[label_item] += 1

    # 计算指标
    overall_accuracy = total_correct / total_samples
    logger.info(f"测试准确率: {overall_accuracy:.4f}")

    # 每类准确率
    logger.info("各类别准确率:")
    for class_idx in sorted(class_total.keys()):
        if class_total[class_idx] > 0:
            acc = class_correct[class_idx] / class_total[class_idx]
            category_name = MultiModalCADDataset.label_to_category(class_idx)
            logger.info(f"  {category_name}: {acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

    return overall_accuracy


def main():
    parser = argparse.ArgumentParser(description="CADNET多模态分类训练")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=r"D:\PythonProjects\Network\data\annotations\cadnet_annotations.json",
        help="标注文件路径"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="实验名称"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="仅测试模式"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="测试模式下的模型路径"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复训练的路径"
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default=None,
        help="TensorBoard日志目录"
    )

    args = parser.parse_args()

    # 创建实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"cadnet_classification_{timestamp}"

    # 配置
    config = BaseConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.resume_from_checkpoint = args.resume_from_checkpoint

    # 设置TensorBoard日志目录
    if args.tensorboard_log_dir:
        config.tensorboard_log_dir = args.tensorboard_log_dir

    model_config = ModelConfig()
    model_config.num_classes = 43  # CADNET 43个类别

    # 设置日志
    logger = setup_logging(config.log_dir, args.experiment_name)

    # 设置随机种子
    set_random_seed(config.seed)

    # 检查标注文件
    if not os.path.exists(args.annotation_file):
        logger.error(f"标注文件不存在: {args.annotation_file}")
        logger.info("请先运行: python generate_annotations.py")
        return

    if args.test_only:
        # 仅测试模式
        if not args.model_path or not os.path.exists(args.model_path):
            logger.error(f"模型文件不存在: {args.model_path}")
            return

        # 创建测试数据集
        _, _, dataset = create_dataloaders(args.annotation_file, config)
        test_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=multimodal_collate_fn
        )

        # 测试
        test_accuracy = test_model(args.model_path, test_loader, logger)
        logger.info(f"最终测试准确率: {test_accuracy:.4f}")

    else:
        # 训练模式
        logger.info(f"开始实验: {args.experiment_name}")
        logger.info(f"配置: {config}")
        logger.info(f"模型配置: {model_config}")

        # 创建数据加载器
        train_loader, val_loader, dataset = create_dataloaders(args.annotation_file, config)

        # 训练模型
        model, best_accuracy = train_model(
            config, model_config, train_loader, val_loader,
            args.experiment_name, logger
        )

        logger.info(f"实验完成: {args.experiment_name}")
        logger.info(f"最佳验证准确率: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()