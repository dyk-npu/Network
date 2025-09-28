#!/usr/bin/env python3
"""
CADNET多模态检索模型训练脚本
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
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.data.multimodal_dataset import MultiModalCADDataset, multimodal_collate_fn
from src.models.downstream.retrieval_pipeline import (
    create_retrieval_pipeline,
    RetrievalTrainer
)


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
        transform=None
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
        pin_memory=True,
        drop_last=True  # 确保批次大小一致，对检索任务很重要
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return train_loader, val_loader, dataset


def train_model(config: BaseConfig, model_config: ModelConfig, train_loader, val_loader,
                experiment_name: str, logger):
    """训练检索模型"""

    # 创建模型
    logger.info("初始化检索模型...")
    model = create_retrieval_pipeline(model_config)

    # 创建训练器
    trainer = RetrievalTrainer(model, model_config, config)

    # 初始化训练参数
    start_epoch = 0
    best_map = 0.0

    # 恢复训练（如果指定了检查点）
    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        logger.info(f"从检查点恢复训练: {config.resume_from_checkpoint}")

        # 检查是否为分类模型检查点
        checkpoint = torch.load(config.resume_from_checkpoint, map_location='cpu')
        checkpoint_keys = list(checkpoint['model_state_dict'].keys())

        # 如果包含分类器组件，则认为是分类检查点
        is_classification_checkpoint = any(key.startswith('classifier.') for key in checkpoint_keys)

        if is_classification_checkpoint:
            logger.info("检测到分类模型检查点，加载兼容组件...")
            trainer.load_classification_checkpoint(config.resume_from_checkpoint)
            # 从头开始训练检索任务
            start_epoch = 0
            best_map = 0.0
        else:
            # 检索模型检查点，正常恢复训练
            start_epoch, best_map = trainer.load_checkpoint(config.resume_from_checkpoint)
            logger.info(f"恢复检索训练从epoch {start_epoch + 1}, 最佳mAP: {best_map:.4f}")

        del checkpoint  # 释放内存

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
    patience = 15  # 检索任务可能需要更多耐心
    patience_counter = 0

    logger.info("开始训练检索模型...")

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        # 训练一个epoch
        train_metrics = trainer.train_epoch(train_loader)

        # 记录训练指标
        writer.add_scalar("Train/loss", train_metrics['loss'], epoch)
        writer.add_scalar("Train/learning_rate", train_metrics['learning_rate'], epoch)

        logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                   f"LR: {train_metrics['learning_rate']:.6f}")

        # 验证
        if (epoch + 1) % config.eval_interval == 0:
            val_metrics = trainer.validate(val_loader)

            # 记录验证指标
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Val/{key}", value, epoch)

            # 记录主要指标
            current_map = val_metrics.get('mAP', 0.0)
            recall_at_10 = val_metrics.get('Recall@10', 0.0)

            logger.info(f"验证 - mAP: {current_map:.4f}, "
                       f"Recall@1: {val_metrics.get('Recall@1', 0.0):.4f}, "
                       f"Recall@5: {val_metrics.get('Recall@5', 0.0):.4f}, "
                       f"Recall@10: {recall_at_10:.4f}")

            # 保存最佳模型
            if current_map > best_map:
                best_map = current_map
                patience_counter = 0

                model_save_path = os.path.join(
                    config.model_save_dir, experiment_name,
                    f"best_model_map_{current_map:.4f}.pth"
                )
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                trainer.save_checkpoint(model_save_path, epoch, best_map, is_best=True)
                logger.info(f"保存最佳模型: {model_save_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"mAP连续{patience}个epoch没有提升，提前停止训练")
                break

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.model_save_dir, experiment_name,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            trainer.save_checkpoint(checkpoint_path, epoch, best_map)
            logger.info(f"保存检查点: {checkpoint_path}")

    writer.close()
    logger.info(f"训练完成! 最佳mAP: {best_map:.4f}")

    return model, best_map


def test_model(model_path: str, test_loader, logger):
    """测试检索模型"""
    logger.info(f"加载检索模型: {model_path}")

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']

    model = create_retrieval_pipeline(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # 构建测试候选集
    logger.info("构建测试候选集...")
    all_images = []
    all_brep_graphs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="构建候选集"):
            all_images.append(batch['images'])
            all_brep_graphs.append(batch['brep_graph'])
            all_labels.append(batch['labels'])

    # 拼接所有数据
    candidate_images = torch.cat(all_images, dim=0).to(device)
    candidate_brep = torch.cat(all_brep_graphs, dim=0).to(device)
    candidate_labels = torch.cat(all_labels, dim=0)

    logger.info(f"候选集大小: {len(candidate_labels)}")

    # 执行检索测试
    from src.models.downstream.retrieval import RetrievalMetrics
    metrics = RetrievalMetrics()

    logger.info("执行检索测试...")
    test_batch_size = 32  # 测试时使用较小的批次

    with torch.no_grad():
        for i in tqdm(range(0, len(candidate_images), test_batch_size), desc="检索测试"):
            end_idx = min(i + test_batch_size, len(candidate_images))
            query_images = candidate_images[i:end_idx]
            query_brep = candidate_brep[i:end_idx]
            query_labels = candidate_labels[i:end_idx]

            # 执行检索
            results = model.retrieve(
                query_images, query_brep,
                candidate_images, candidate_brep,
                k=20, modality='fused'
            )

            # 更新指标
            metrics.update(results['indices'], candidate_labels, query_labels)

    # 计算最终指标
    final_metrics = metrics.compute()

    logger.info("检索测试结果:")
    for metric_name, value in final_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="CADNET多模态检索训练")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=r"D:\PythonProjects\Network\letters_annotations.json",
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
        default=16,  # 检索任务使用较小的批次大小
        help="批大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,  # 检索任务通常需要较少的epoch
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,  # 使用较小的学习率
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
        args.experiment_name = f"cadnet_retrieval_{timestamp}"

    # 配置
    config = BaseConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.resume_from_checkpoint = args.resume_from_checkpoint
    config.eval_interval = 2  # 检索任务更频繁的验证

    # 设置TensorBoard日志目录
    if args.tensorboard_log_dir:
        config.tensorboard_log_dir = args.tensorboard_log_dir

    model_config = ModelConfig()
    # 添加检索相关配置
    model_config.retrieval_dim = 256

    # 设置日志
    logger = setup_logging(config.log_dir, args.experiment_name)

    # 设置随机种子
    set_random_seed(config.seed)

    # 检查标注文件
    if not os.path.exists(args.annotation_file):
        logger.error(f"标注文件不存在: {args.annotation_file}")
        logger.info("请先运行字母数据集生成脚本")
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
        test_metrics = test_model(args.model_path, test_loader, logger)
        logger.info(f"最终测试结果: mAP = {test_metrics.get('mAP', 0.0):.4f}")

    else:
        # 训练模式
        logger.info(f"开始检索模型训练实验: {args.experiment_name}")
        logger.info(f"配置: {config}")
        logger.info(f"模型配置: {model_config}")

        # 创建数据加载器
        train_loader, val_loader, dataset = create_dataloaders(args.annotation_file, config)

        # 训练模型
        model, best_map = train_model(
            config, model_config, train_loader, val_loader,
            args.experiment_name, logger
        )

        logger.info(f"实验完成: {args.experiment_name}")
        logger.info(f"最佳mAP: {best_map:.4f}")


if __name__ == "__main__":
    main()