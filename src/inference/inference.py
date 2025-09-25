#!/usr/bin/env python3
"""
CADNET模型推理脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import dgl
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from config.model_config import ModelConfig
from src.models.downstream.classification_pipeline import MultiModalClassificationPipeline
from src.data.multimodal_dataset import MultiModalCADDataset


def load_model(model_path: str):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    model = MultiModalClassificationPipeline(config)

    # 处理适应性层的状态字典加载问题
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()

    # 找出checkpoint中存在但model中不存在的adaptive层
    adaptive_keys = [key for key in state_dict.keys() if 'adaptive_' in key]

    if adaptive_keys:
        print(f"发现 {len(adaptive_keys)} 个适应性层，正在重新创建...")

        # 为每个adaptive层预先创建对应的层
        for key in adaptive_keys:
            if key.startswith('brep_encoder.adaptive_'):
                # 解析层名称获取输入输出维度
                layer_name = key.replace('brep_encoder.adaptive_', '').replace('.weight', '').replace('.bias', '')
                if layer_name and '_' in layer_name:
                    dims = layer_name.split('_')
                    if len(dims) >= 2:
                        try:
                            input_dim = int(dims[0])
                            output_dim = int(dims[1])

                            # 创建适应性投影层
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            adaptive_layer = model.brep_encoder._get_adaptive_projection(
                                input_dim, output_dim, device='cpu'
                            )
                            print(f"  创建适应性层: {input_dim} -> {output_dim}")
                        except ValueError:
                            print(f"  警告: 无法解析适应性层维度: {layer_name}")

    # 现在尝试加载状态字典
    try:
        model.load_state_dict(state_dict, strict=True)
        print("状态字典加载成功")
    except RuntimeError as e:
        print(f"严格模式加载失败: {e}")
        # 尝试非严格模式加载
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"模型加载完成，设备: {device}")
    return model, device, config


def preprocess_images(image_paths: list, target_size=(224, 224)):
    """预处理图像"""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            image = Image.open(path).convert('RGB')
            image = image.resize(target_size, Image.LANCZOS)  # 使用高质量重采样
            # 转换为tensor并归一化
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            images.append(image_tensor)
        else:
            # 如果图像不存在，创建空白图像
            blank_image = torch.ones(3, target_size[0], target_size[1])
            images.append(blank_image)

    return torch.stack(images).unsqueeze(0)  # [1, num_views, C, H, W]


def load_brep_graph(brep_path: str):
    """加载B-rep图"""
    if os.path.exists(brep_path):
        graphs, _ = dgl.data.utils.load_graphs(brep_path)
        graph = graphs[0]

        # 确保图的特征是Float类型，避免与模型权重类型不匹配
        if 'x' in graph.ndata:
            graph.ndata['x'] = graph.ndata['x'].float()
        if 'x' in graph.edata:
            graph.edata['x'] = graph.edata['x'].float()

        return graph
    else:
        print(f"警告: B-rep文件不存在: {brep_path}")
        return None


def predict_single_model(model, device, image_paths: list, brep_path: str):
    """对单个模型进行预测"""
    # 预处理输入
    images = preprocess_images(image_paths).to(device)

    brep_graph = load_brep_graph(brep_path)
    if brep_graph is None:
        print("无法加载B-rep图，跳过预测")
        return None

    graphs = dgl.batch([brep_graph]).to(device)

    # 预测
    with torch.no_grad():
        results = model.predict(images, graphs)

    return {
        'prediction': results['predictions'].cpu().item(),
        'confidence': results['confidence'].cpu().item(),
        'probabilities': results['probabilities'].cpu().numpy().flatten()
    }


def predict_from_annotation(model, device, annotation_file: str, item_id: str):
    """从标注文件中预测指定模型"""
    import json

    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    if item_id not in annotations:
        print(f"模型ID {item_id} 不存在于标注文件中")
        return None

    annotation = annotations[item_id]
    available_data = annotation.get('available_data', {})

    # 获取图像路径
    views = available_data.get('views', {})
    image_paths = [
        views.get('front', ''),
        views.get('side', ''),
        views.get('top', '')
    ]

    # 获取B-rep路径
    brep_path = available_data.get('brep', '')

    # 预测
    result = predict_single_model(model, device, image_paths, brep_path)

    if result:
        # 获取真实类别
        true_category = annotation.get('metadata', {}).get('category', 'unknown')
        predicted_category = MultiModalCADDataset.label_to_category(result['prediction'])

        print(f"\n预测结果:")
        print(f"  模型ID: {item_id}")
        print(f"  真实类别: {true_category}")
        print(f"  预测类别: {predicted_category}")
        print(f"  预测置信度: {result['confidence']:.4f}")
        print(f"  预测正确: {'✅' if true_category == predicted_category else '❌'}")

        # 显示top-5预测
        top5_indices = np.argsort(result['probabilities'])[-5:][::-1]
        print(f"  Top-5预测:")
        for i, idx in enumerate(top5_indices):
            category = MultiModalCADDataset.label_to_category(idx)
            prob = result['probabilities'][idx]
            print(f"    {i+1}. {category}: {prob:.4f}")

    return result


def batch_inference(model, device, annotation_file: str, num_samples: int = 10):
    """批量推理测试"""
    import json

    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 获取所有模型ID（排除元信息）
    model_ids = [k for k in annotations.keys() if not k.startswith('_')]

    # 随机选择样本
    import random
    random.seed(42)
    selected_ids = random.sample(model_ids, min(num_samples, len(model_ids)))

    print(f"\n批量推理测试 ({len(selected_ids)} 个样本):")
    print("=" * 60)

    correct_predictions = 0
    total_predictions = 0

    for item_id in selected_ids:
        result = predict_from_annotation(model, device, annotation_file, item_id)
        if result:
            annotation = annotations[item_id]
            true_category = annotation.get('metadata', {}).get('category', 'unknown')
            predicted_category = MultiModalCADDataset.label_to_category(result['prediction'])

            if true_category == predicted_category:
                correct_predictions += 1
            total_predictions += 1

        print("-" * 40)

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n批量推理准确率: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    else:
        print("\n无有效预测结果")


def main():
    parser = argparse.ArgumentParser(description="CADNET模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="cadnet_annotations.json",
        help="标注文件路径"
    )
    parser.add_argument(
        "--item_id",
        type=str,
        help="要预测的模型ID"
    )
    parser.add_argument(
        "--batch_test",
        type=int,
        default=0,
        help="批量测试样本数量"
    )
    parser.add_argument(
        "--image_paths",
        nargs=3,
        help="直接指定图像路径 [front, side, top]"
    )
    parser.add_argument(
        "--brep_path",
        type=str,
        help="B-rep文件路径"
    )

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        return

    if not os.path.exists(args.annotation_file):
        print(f"标注文件不存在: {args.annotation_file}")
        return

    # 加载模型
    model, device, config = load_model(args.model_path)

    if args.batch_test > 0:
        # 批量测试
        batch_inference(model, device, args.annotation_file, args.batch_test)

    elif args.item_id:
        # 单个模型预测（从标注文件）
        predict_from_annotation(model, device, args.annotation_file, args.item_id)

    elif args.image_paths and args.brep_path:
        # 直接指定文件路径预测
        result = predict_single_model(model, device, args.image_paths, args.brep_path)
        if result:
            predicted_category = MultiModalCADDataset.label_to_category(result['prediction'])
            print(f"\n预测结果:")
            print(f"  预测类别: {predicted_category}")
            print(f"  置信度: {result['confidence']:.4f}")

    else:
        print("请指定预测方式:")
        print("  --item_id MODEL_ID        # 从标注文件预测单个模型")
        print("  --batch_test N            # 批量测试N个样本")
        print("  --image_paths + --brep_path # 直接指定文件路径")


if __name__ == "__main__":
    main()