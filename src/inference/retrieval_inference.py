#!/usr/bin/env python3
"""
CADNET检索模型推理脚本 - 支持多模态相似性检索
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import dgl
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.model_config import ModelConfig
from src.models.downstream.retrieval_pipeline import create_retrieval_pipeline
from src.data.multimodal_dataset import MultiModalCADDataset


def load_retrieval_model(model_path: str):
    """加载检索模型"""
    print(f"加载检索模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # 获取配置
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("警告: 使用默认配置")
        config = ModelConfig()

    # 创建模型
    model = create_retrieval_pipeline(config)

    # 加载状态字典
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ 模型状态字典加载成功")
    except RuntimeError as e:
        print(f"严格模式加载失败: {e}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"缺失的键: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        print("✓ 模型状态字典加载成功（非严格模式）")

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"检索模型加载完成，设备: {device}")
    return model, device, config


def preprocess_images(image_paths: List[str], target_size=(224, 224)):
    """预处理图像数据"""
    images = []
    valid_count = 0

    # 确保有3个视图（不足的用空白填充）
    while len(image_paths) < 3:
        image_paths.append(None)

    for i, path in enumerate(image_paths[:3]):  # 只取前3个
        if path and os.path.exists(path):
            try:
                image = Image.open(path).convert('RGB')
                image = image.resize(target_size, Image.LANCZOS)

                # 转换为tensor并归一化
                image_array = np.array(image).astype(np.float32) / 255.0

                # 标准化 (ImageNet标准)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_array = (image_array - mean) / std

                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                images.append(image_tensor)
                valid_count += 1
                print(f"✓ 加载图像 {i+1}: {os.path.basename(path)}")
            except Exception as e:
                print(f"✗ 图像加载失败 {i+1}: {e}")
                blank_image = torch.zeros(3, target_size[0], target_size[1])
                images.append(blank_image)
        else:
            print(f"✗ 图像路径为空或不存在 {i+1}")
            blank_image = torch.zeros(3, target_size[0], target_size[1])
            images.append(blank_image)

    if valid_count == 0:
        return None

    print(f"✓ 成功处理 {valid_count}/3 个图像")
    return torch.stack(images).unsqueeze(0)  # [1, 3, C, H, W]


def load_brep_graph(brep_path: str):
    """加载Brep图数据"""
    if not brep_path or not os.path.exists(brep_path):
        return None

    try:
        graphs, _ = dgl.data.utils.load_graphs(brep_path)
        graph = graphs[0]

        # 确保特征类型正确
        if 'x' in graph.ndata:
            graph.ndata['x'] = graph.ndata['x'].float()
        if 'x' in graph.edata:
            graph.edata['x'] = graph.edata['x'].float()

        print(f"✓ 加载Brep图: {os.path.basename(brep_path)}")
        print(f"  节点数: {graph.num_nodes()}, 边数: {graph.num_edges()}")
        return graph

    except Exception as e:
        print(f"✗ Brep图加载失败: {e}")
        return None


def build_candidate_database(model, device, annotation_file: str, max_candidates: int = 1000):
    """构建候选数据库"""
    print(f"\n📚 构建候选数据库...")

    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 获取所有模型ID
    model_ids = [k for k in annotations.keys() if not k.startswith('_')]
    if len(model_ids) > max_candidates:
        import random
        random.seed(42)
        model_ids = random.sample(model_ids, max_candidates)

    print(f"处理 {len(model_ids)} 个候选样本...")

    candidate_features = []
    candidate_metadata = []
    processed_count = 0

    for i, model_id in enumerate(model_ids):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(model_ids)}")

        annotation = annotations[model_id]
        available_data = annotation.get('available_data', {})

        # 获取图像路径
        views = available_data.get('views', {})
        image_paths = [
            views.get('front', ''),
            views.get('side', ''),
            views.get('top', '')
        ]

        # 获取Brep路径
        brep_path = available_data.get('brep', '')

        # 预处理输入
        images = preprocess_images(image_paths) if any(image_paths) else None
        brep_graph = load_brep_graph(brep_path) if brep_path else None

        if images is None and brep_graph is None:
            continue

        try:
            # 编码特征
            with torch.no_grad():
                if images is not None:
                    images = images.to(device)
                if brep_graph is not None:
                    brep_graph = dgl.batch([brep_graph]).to(device)

                features = model.encode_features(
                    images if images is not None else torch.zeros(1, 3, 3, 224, 224).to(device),
                    brep_graph if brep_graph is not None else None
                )

                candidate_features.append({
                    'fused_features': features['fused_features'].cpu(),
                    'image_enhanced': features['image_enhanced'].cpu(),
                    'brep_enhanced': features['brep_enhanced'].cpu()
                })

                candidate_metadata.append({
                    'model_id': model_id,
                    'category': annotation.get('metadata', {}).get('category', 'unknown'),
                    'has_images': images is not None,
                    'has_brep': brep_graph is not None
                })

                processed_count += 1

        except Exception as e:
            print(f"  跳过 {model_id}: {e}")
            continue

    print(f"✓ 成功构建候选数据库: {processed_count} 个样本")

    # 转换为批量张量
    if candidate_features:
        candidate_batch = {
            'fused_features': torch.cat([f['fused_features'] for f in candidate_features], dim=0),
            'image_enhanced': torch.cat([f['image_enhanced'] for f in candidate_features], dim=0),
            'brep_enhanced': torch.cat([f['brep_enhanced'] for f in candidate_features], dim=0)
        }
    else:
        candidate_batch = None

    return candidate_batch, candidate_metadata


def retrieve_similar(model, device, query_images=None, query_brep=None,
                    candidate_features=None, candidate_metadata=None,
                    k: int = 10, modality: str = 'fused'):
    """执行相似性检索"""
    print(f"\n🔍 执行检索...")
    print(f"  检索模态: {modality}")
    print(f"  返回数量: Top-{k}")

    if candidate_features is None or not candidate_metadata:
        print("❌ 候选数据库为空")
        return None

    # 编码查询特征
    with torch.no_grad():
        if query_images is not None:
            query_images = query_images.to(device)
        if query_brep is not None:
            query_brep = dgl.batch([query_brep]).to(device)

        query_features = model.encode_features(
            query_images if query_images is not None else torch.zeros(1, 3, 3, 224, 224).to(device),
            query_brep if query_brep is not None else None
        )

        # 移动候选特征到设备
        candidate_batch = {
            key: features.to(device)
            for key, features in candidate_features.items()
        }

        # 执行检索
        results = model.retrieval_module.retrieve(
            query_features, candidate_batch, k=k, modality=modality
        )

    # 处理结果
    similarities = results['similarities'].cpu().numpy().flatten()
    indices = results['indices'].cpu().numpy().flatten()

    retrieved_results = []
    for i, (idx, sim) in enumerate(zip(indices, similarities)):
        if idx < len(candidate_metadata):
            metadata = candidate_metadata[idx]
            retrieved_results.append({
                'rank': i + 1,
                'model_id': metadata['model_id'],
                'category': metadata['category'],
                'similarity': float(sim),
                'has_images': metadata['has_images'],
                'has_brep': metadata['has_brep']
            })

    return retrieved_results


def retrieve_from_query(model, device, query_images=None, query_brep=None,
                       candidate_features=None, candidate_metadata=None,
                       k: int = 10):
    """从查询执行检索"""
    if query_images is None and query_brep is None:
        print("❌ 至少需要提供一种查询模态")
        return None

    # 确定查询模态
    if query_images is not None and query_brep is not None:
        query_modality = "dual"
        modalities = ['fused', 'image', 'brep']
    elif query_images is not None:
        query_modality = "image"
        modalities = ['image', 'fused']
    else:
        query_modality = "brep"
        modalities = ['brep', 'fused']

    print(f"\n🎯 查询模态: {query_modality}")

    all_results = {}
    for modality in modalities:
        print(f"\n--- 使用 {modality} 特征检索 ---")
        results = retrieve_similar(
            model, device, query_images, query_brep,
            candidate_features, candidate_metadata, k, modality
        )
        if results:
            all_results[modality] = results

    return all_results


def display_retrieval_results(results: Dict[str, List[Dict]], top_n: int = 5):
    """显示检索结果"""
    print(f"\n📊 检索结果 (Top-{top_n}):")
    print("=" * 80)

    for modality, result_list in results.items():
        print(f"\n🔹 {modality.upper()} 特征检索:")
        print("-" * 50)

        for i, result in enumerate(result_list[:top_n]):
            print(f"  {result['rank']:2d}. {result['model_id']:<15} "
                  f"类别: {result['category']:<12} "
                  f"相似度: {result['similarity']:.4f} "
                  f"{'📷' if result['has_images'] else '❌'}"
                  f"{'🔧' if result['has_brep'] else '❌'}")


def retrieve_from_annotation(model, device, annotation_file: str, query_id: str,
                            candidate_features=None, candidate_metadata=None, k: int = 10):
    """从标注文件检索"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    if query_id not in annotations:
        print(f"❌ 查询ID {query_id} 不存在")
        return None

    query_annotation = annotations[query_id]
    available_data = query_annotation.get('available_data', {})

    print(f"\n🔎 检索查询: {query_id}")
    print(f"  真实类别: {query_annotation.get('metadata', {}).get('category', 'unknown')}")

    # 获取查询数据
    views = available_data.get('views', {})
    image_paths = [views.get('front', ''), views.get('side', ''), views.get('top', '')]
    brep_path = available_data.get('brep', '')

    # 预处理查询输入
    query_images = preprocess_images(image_paths) if any(image_paths) else None
    query_brep = load_brep_graph(brep_path) if brep_path else None

    if query_images is None and query_brep is None:
        print("❌ 查询样本没有有效数据")
        return None

    # 执行检索
    results = retrieve_from_query(
        model, device, query_images, query_brep,
        candidate_features, candidate_metadata, k
    )

    if results:
        display_retrieval_results(results, top_n=k)

        # 计算检索准确率
        true_category = query_annotation.get('metadata', {}).get('category', 'unknown')
        print(f"\n📈 检索准确率分析:")

        for modality, result_list in results.items():
            correct_at_k = []
            for k_val in [1, 5, 10]:
                if k_val <= len(result_list):
                    top_k_categories = [r['category'] for r in result_list[:k_val]]
                    correct = true_category in top_k_categories
                    correct_at_k.append(f"R@{k_val}: {'✅' if correct else '❌'}")

            print(f"  {modality}: {' | '.join(correct_at_k)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CADNET检索模型推理")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的检索模型路径"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="letters_annotations.json",
        help="标注文件路径"
    )
    parser.add_argument(
        "--query_id",
        type=str,
        help="查询模型ID（从标注文件）"
    )
    parser.add_argument(
        "--query_images",
        nargs='*',
        help="查询图像路径列表"
    )
    parser.add_argument(
        "--query_brep",
        type=str,
        help="查询Brep文件路径"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="返回Top-K结果数量"
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=1000,
        help="最大候选样本数量"
    )
    parser.add_argument(
        "--modality",
        choices=['image', 'brep', 'fused', 'all'],
        default='all',
        help="检索使用的特征模态"
    )

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return

    if not os.path.exists(args.annotation_file):
        print(f"❌ 标注文件不存在: {args.annotation_file}")
        return

    # 加载模型
    try:
        model, device, config = load_retrieval_model(args.model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 构建候选数据库
    try:
        candidate_features, candidate_metadata = build_candidate_database(
            model, device, args.annotation_file, args.max_candidates
        )
        if candidate_features is None:
            print("❌ 候选数据库构建失败")
            return
    except Exception as e:
        print(f"❌ 候选数据库构建失败: {e}")
        return

    # 执行检索
    if args.query_id:
        # 从标注文件检索
        retrieve_from_annotation(
            model, device, args.annotation_file, args.query_id,
            candidate_features, candidate_metadata, args.top_k
        )

    elif args.query_images or args.query_brep:
        # 直接文件路径检索
        query_images = preprocess_images(args.query_images) if args.query_images else None
        query_brep = load_brep_graph(args.query_brep) if args.query_brep else None

        if query_images is None and query_brep is None:
            print("❌ 没有有效的查询输入")
            return

        results = retrieve_from_query(
            model, device, query_images, query_brep,
            candidate_features, candidate_metadata, args.top_k
        )

        if results:
            display_retrieval_results(results, args.top_k)

    else:
        print("❌ 请指定查询方式:")
        print("  --query_id MODEL_ID                    # 从标注文件检索")
        print("  --query_images path1 [path2 path3]     # 图像查询")
        print("  --query_brep PATH                      # Brep查询")
        print("  --query_images ... --query_brep ...    # 多模态查询")
        print("")
        print("📝 使用示例:")
        print("  # 从标注文件检索")
        print("  python retrieval_inference.py --model_path model.pth --query_id MODEL_001")
        print("")
        print("  # 图像检索")
        print("  python retrieval_inference.py --model_path model.pth --query_images front.jpg side.jpg")
        print("")
        print("  # Brep检索")
        print("  python retrieval_inference.py --model_path model.pth --query_brep query.dgl")


if __name__ == "__main__":
    main()