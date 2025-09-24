#!/usr/bin/env python3
"""
测试多模态分类功能
"""

import torch
import dgl
import numpy as np
from config.model_config import ModelConfig
from src.models.downstream import (
    create_classification_model,
    create_classification_loss,
    create_classification_pipeline,
    ClassificationMetrics,
    CAD_PART_CLASSES
)


def test_classification_model():
    """测试分类模型"""
    print("Testing Classification Model...")

    try:
        config = ModelConfig()
        classifier = create_classification_model(config)
        criterion = create_classification_loss(config)

        batch_size = 4

        # 模拟融合特征输入
        fusion_output = {
            'fused_features': torch.randn(batch_size, config.fusion_dim),
            'image_enhanced': torch.randn(batch_size, config.fusion_dim),
            'brep_enhanced': torch.randn(batch_size, config.fusion_dim)
        }

        # 模拟标签
        targets = torch.randint(0, config.num_classes, (batch_size,))

        with torch.no_grad():
            # 测试前向传播
            outputs = classifier(fusion_output)
            losses = criterion(outputs, targets)

            print(f"  ✓ 主分类器输出形状: {outputs['logits'].shape}")
            print(f"  ✓ 概率输出形状: {outputs['probabilities'].shape}")
            print(f"  ✓ 特征重要性形状: {outputs['feature_importance'].shape}")
            print(f"  ✓ 总损失: {losses['total_loss'].item():.4f}")

            # 测试预测功能
            predictions = classifier.predict(fusion_output)
            print(f"  ✓ 预测结果: {predictions['predictions']}")
            print(f"  ✓ 置信度: {predictions['confidence']}")

            # 验证输出维度
            assert outputs['logits'].shape == (batch_size, config.num_classes)
            assert outputs['probabilities'].shape == (batch_size, config.num_classes)
            assert outputs['feature_importance'].shape == (batch_size, 1)

        print("  ✓ Classification model test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Classification model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classification_metrics():
    """测试分类指标"""
    print("Testing Classification Metrics...")

    try:
        config = ModelConfig()
        metrics = ClassificationMetrics(config.num_classes)

        # 模拟预测和真实标签
        batch_size = 20
        predictions = torch.randint(0, config.num_classes, (batch_size,))
        targets = torch.randint(0, config.num_classes, (batch_size,))

        # 更新指标
        metrics.update(predictions, targets)

        # 计算指标
        results = metrics.compute()

        print(f"  ✓ 整体准确率: {results['accuracy']:.4f}")
        print(f"  ✓ 平均准确率: {results['mean_accuracy']:.4f}")
        print(f"  ✓ 每类准确率数量: {len(results['class_accuracies'])}")

        # 验证指标范围
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['mean_accuracy'] <= 1
        assert len(results['class_accuracies']) == config.num_classes

        print("  ✓ Classification metrics test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Classification metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_classification_pipeline():
    """测试完整分类管道"""
    print("Testing Classification Pipeline...")

    try:
        config = ModelConfig()
        pipeline = create_classification_pipeline(config)

        batch_size = 2

        # 准备图像数据
        images = torch.randn(batch_size, 3, 3, 224, 224)

        # 准备Brep数据
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

        with torch.no_grad():
            # 测试训练模式
            results = pipeline(images, batched_graphs, labels)

            print(f"  ✓ 预测结果形状: {results['predictions'].shape}")
            print(f"  ✓ 概率输出形状: {results['probabilities'].shape}")
            print(f"  ✓ 总损失: {results['total_loss'].item():.4f}")
            print(f"  ✓ 主损失: {results['main_loss'].item():.4f}")

            # 测试推理模式
            pred_results = pipeline.predict(images, batched_graphs)
            print(f"  ✓ 推理预测: {pred_results['predictions']}")
            print(f"  ✓ 置信度范围: [{pred_results['confidence'].min().item():.4f}, {pred_results['confidence'].max().item():.4f}]")

            # 测试特征提取
            features = pipeline.extract_features(images, batched_graphs)
            print(f"  ✓ 图像特征形状: {features['image_features'].shape}")
            print(f"  ✓ Brep特征形状: {features['brep_features'].shape}")
            print(f"  ✓ 融合特征形状: {features['fused_features'].shape}")

            # 验证输出维度
            assert results['predictions'].shape == (batch_size,)
            assert results['probabilities'].shape == (batch_size, config.num_classes)
            assert features['fused_features'].shape == (batch_size, config.fusion_dim)

        print("  ✓ Classification pipeline test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Classification pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cad_part_classes():
    """测试CAD零件类别定义"""
    print("Testing CAD Part Classes...")

    try:
        print(f"  ✓ 定义的CAD零件类别数量: {len(CAD_PART_CLASSES)}")
        print(f"  ✓ 类别列表: {CAD_PART_CLASSES}")

        # 验证类别数量与配置一致
        config = ModelConfig()
        assert len(CAD_PART_CLASSES) == config.num_classes, f"类别数量不匹配: {len(CAD_PART_CLASSES)} != {config.num_classes}"

        print("  ✓ CAD part classes test passed!")
        return True

    except Exception as e:
        print(f"  ✗ CAD part classes test failed: {e}")
        return False


def test_end_to_end_classification():
    """端到端分类测试"""
    print("Testing End-to-End Classification...")

    try:
        config = ModelConfig()
        pipeline = create_classification_pipeline(config)

        batch_size = 3

        # 创建更真实的测试数据
        images = torch.randn(batch_size, 3, 3, 224, 224)

        # 创建不同大小的图
        graphs = []
        expected_predictions = []

        for i in range(batch_size):
            # 不同的图大小
            num_nodes = 4 + i * 2

            # 创建连通图
            edges = []
            for j in range(num_nodes):
                edges.append((j, (j + 1) % num_nodes))  # 环形连接

            if num_nodes > 3:  # 添加额外连接
                edges.append((0, num_nodes // 2))
                edges.append((1, num_nodes - 1))

            src, dst = zip(*edges)
            graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
            graph = dgl.add_self_loop(graph)

            # 不同的特征模式
            if i == 0:  # 简单特征
                graph.ndata['x'] = torch.ones(num_nodes, 7) * 0.5
            elif i == 1:  # 随机特征
                graph.ndata['x'] = torch.randn(num_nodes, 7)
            else:  # 结构化特征
                features = torch.zeros(num_nodes, 7)
                features[:, i % 7] = 1.0  # one-hot like
                graph.ndata['x'] = features

            graphs.append(graph)

        batched_graphs = dgl.batch(graphs)

        with torch.no_grad():
            # 推理测试
            results = pipeline.predict(images, batched_graphs)

            print(f"  ✓ 批量预测结果: {results['predictions']}")
            print(f"  ✓ 平均置信度: {results['confidence'].mean().item():.4f}")
            print(f"  ✓ 特征重要性均值: {results['feature_importance'].mean().item():.4f}")

            # 单独预测每个样本
            for i in range(batch_size):
                single_image = images[i:i+1]
                single_graph = dgl.unbatch(batched_graphs)[i]
                single_batched = dgl.batch([single_graph])

                single_result = pipeline.predict(single_image, single_batched)
                print(f"  ✓ 样本 {i+1} 预测: 类别 {single_result['predictions'].item()}, 置信度 {single_result['confidence'].item():.4f}")

        print("  ✓ End-to-end classification test passed!")
        return True

    except Exception as e:
        print(f"  ✗ End-to-end classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("TESTING MULTIMODAL CLASSIFICATION")
    print("=" * 60)

    tests = [
        ("Classification Model", test_classification_model),
        ("Classification Metrics", test_classification_metrics),
        ("Classification Pipeline", test_classification_pipeline),
        ("CAD Part Classes", test_cad_part_classes),
        ("End-to-End Classification", test_end_to_end_classification)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
        print()

    # 总结
    print("=" * 60)
    print("CLASSIFICATION TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if not success:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("🎉 ALL CLASSIFICATION TESTS PASSED!")
        print("The multimodal classification system is working correctly.")
        print(f"✓ Support for {len(CAD_PART_CLASSES)} CAD part classes")
        print("✓ End-to-end pipeline ready for training")
    else:
        print("❌ Some classification tests failed. Please check the error messages above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)