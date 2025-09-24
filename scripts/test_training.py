#!/usr/bin/env python3
"""
测试训练流程 - 小批量数据快速验证
"""

import os
import sys
import torch
import logging
from torch.utils.data import DataLoader, Subset

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from config.model_config import ModelConfig
from config.base_config import BaseConfig
from src.data.multimodal_dataset import MultiModalCADDataset, multimodal_collate_fn
from src.models.downstream.classification_pipeline import MultiModalClassificationPipeline, ClassificationTrainer


def test_training_pipeline():
    """测试训练管道"""
    print("=" * 60)
    print("测试CADNET训练管道")
    print("=" * 60)

    # 配置
    config = BaseConfig()
    config.batch_size = 4  # 小批量测试
    config.num_epochs = 3  # 少量epoch测试
    config.learning_rate = 1e-4

    model_config = ModelConfig()
    model_config.num_classes = 43

    # 检查标注文件
    annotation_file = r"D:\PythonProjects\Network\data\annotations\cadnet_annotations.json"
    if not os.path.exists(annotation_file):
        print(f"❌ 标注文件不存在: {annotation_file}")
        print("请先运行: python generate_annotations.py")
        return

    try:
        print("\n1. 加载数据集...")
        dataset = MultiModalCADDataset(
            annotation_file=annotation_file,
            modalities=["image", "brep"]
        )
        print(f"   数据集大小: {len(dataset)}")

        # 获取类别统计
        stats = dataset.get_type_statistics()
        print(f"   类别数量: {len(stats['categories'])}")

        # 使用小部分数据测试
        test_size = min(16, len(dataset))
        subset = Subset(dataset, range(test_size))
        print(f"   测试子集大小: {len(subset)}")

        print("\n2. 创建数据加载器...")
        dataloader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=multimodal_collate_fn
        )

        print("\n3. 测试数据加载...")
        for i, batch in enumerate(dataloader):
            print(f"   批次 {i+1}:")
            print(f"     样本数: {len(batch['item_id'])}")
            print(f"     图像形状: {batch['images'].shape if 'images' in batch else 'N/A'}")
            print(f"     B-rep图: {batch['brep_graph'].number_of_nodes() if 'brep_graph' in batch else 'N/A'} 节点")
            print(f"     标签: {batch['labels'].tolist()}")
            print(f"     类别: {batch['type'][:2]}...")

            if i >= 1:  # 只测试前两个批次
                break

        print("\n4. 创建模型...")
        model = MultiModalClassificationPipeline(model_config)
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        print("\n5. 创建训练器...")
        trainer = ClassificationTrainer(model, model_config, config)

        print("\n6. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            images = batch['images'].to(trainer.device)  # 移动到设备
            graphs = batch['brep_graph'].to(trainer.device)  # 移动到设备
            labels = batch['labels'].to(trainer.device)  # 移动到设备

            print(f"   输入形状:")
            print(f"     图像: {images.shape} (设备: {images.device}, 类型: {images.dtype})")
            print(f"     图结构: {graphs.number_of_nodes()} 节点, {graphs.number_of_edges()} 边")
            if 'x' in graphs.ndata:
                print(f"     节点特征: {graphs.ndata['x'].shape} (类型: {graphs.ndata['x'].dtype})")
            if 'x' in graphs.edata:
                print(f"     边特征: {graphs.edata['x'].shape} (类型: {graphs.edata['x'].dtype})")
            print(f"     标签: {labels.shape} (设备: {labels.device})")

            # 分步测试前向传播
            try:
                print("   测试图像编码器...")
                image_features = model.image_encoder(images)
                print(f"     图像特征形状: {image_features.shape}")

                print("   测试B-rep编码器...")
                print(f"     输入图信息:")
                print(f"       批处理图节点总数: {graphs.number_of_nodes()}")
                print(f"       批处理图边总数: {graphs.number_of_edges()}")
                print(f"       批处理大小: {graphs.batch_size}")
                print(f"       批处理图数量: {len(graphs.batch_num_nodes())}")
                print(f"       各图节点数: {graphs.batch_num_nodes().tolist()}")

                brep_output = model.brep_encoder(graphs)
                brep_features = brep_output['features']
                print(f"     B-rep特征形状: {brep_features.shape}")
                print(f"     全局池化特征形状: {brep_output['global_pooled'].shape}")

                print("   测试特征融合...")
                fusion_output = model.fuser(image_features, brep_features)
                print(f"     融合特征形状: {fusion_output['fused_features'].shape}")

                print("   测试分类器...")
                classification_output = model.classifier(fusion_output)
                print(f"     分类输出形状: {classification_output['logits'].shape}")

                print("   完整前向传播...")
                outputs = model(images, graphs, labels)

            except Exception as e:
                print(f"     前向传播失败: {e}")
                # 打印详细信息
                print(f"     图像特征形状: {images.shape}")
                if 'image_features' in locals():
                    print(f"     编码后图像特征: {image_features.shape}")
                if 'brep_features' in locals():
                    print(f"     B-rep特征: {brep_features.shape}")
                raise

            print(f"   输出:")
            print(f"     logits形状: {outputs['logits'].shape}")
            print(f"     预测: {outputs['predictions'].tolist()}")
            print(f"     损失: {outputs['total_loss'].item():.4f}")

        print("\n7. 测试训练一步...")
        model.train()
        train_metrics = {}

        batch = next(iter(dataloader))
        images = batch['images'].to(trainer.device)
        graphs = batch['brep_graph'].to(trainer.device)
        labels = batch['labels'].to(trainer.device)

        # 前向传播
        trainer.optimizer.zero_grad()
        outputs = model(images, graphs, labels)

        # 反向传播
        loss = outputs['total_loss']
        loss.backward()
        trainer.optimizer.step()

        # 统计
        predictions = outputs['predictions']
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.size(0)

        print(f"   一步训练:")
        print(f"     损失: {loss.item():.4f}")
        print(f"     准确率: {accuracy:.4f}")

        print("\n8. 测试完整训练一个epoch...")
        trainer.metrics.reset()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['images'].to(trainer.device)
            graphs = batch['brep_graph'].to(trainer.device)
            labels = batch['labels'].to(trainer.device)

            trainer.optimizer.zero_grad()
            outputs = model(images, graphs, labels)

            loss = outputs['total_loss']
            loss.backward()
            trainer.optimizer.step()

            total_loss += loss.item()
            trainer.metrics.update(outputs['predictions'], labels)

            print(f"     批次 {batch_idx+1}: 损失={loss.item():.4f}")

        # 计算epoch指标
        metrics = trainer.metrics.compute()
        avg_loss = total_loss / len(dataloader)

        print(f"   Epoch结果:")
        print(f"     平均损失: {avg_loss:.4f}")
        print(f"     准确率: {metrics['accuracy']:.4f}")
        print(f"     平均准确率: {metrics['mean_accuracy']:.4f}")

        print("\n✅ 训练管道测试成功!")
        print("现在可以运行完整训练: python train_classification.py")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出

    test_training_pipeline()