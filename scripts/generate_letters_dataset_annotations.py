#!/usr/bin/env python3
"""
生成CADNET数据集的annotation文件
扫描bin和image目录，找到既有bin文件又有三视图的模型
"""

import os
import json
from pathlib import Path
from typing import Dict, Set, List
import argparse


def scan_bin_files(bin_dir: str) -> Dict[str, Set[str]]:
    """
    扫描bin目录，按类别分组文件
    从文件名提取字母和大小写信息作为类别: letter_upper / letter_lower

    Returns:
        Dict[category, Set[filename_without_extension]]
    """
    bin_path = Path(bin_dir)
    bin_files = {}

    print(f"扫描bin目录: {bin_dir}")

    for bin_file in bin_path.glob("*.bin"):
        filename_no_ext = bin_file.stem

        # 解析文件名格式: letter_fontname_case
        parts = filename_no_ext.split('_')
        if len(parts) < 3:
            print(f"  跳过格式不正确的文件: {filename_no_ext}")
            continue

        letter = parts[0]  # 第一部分是字母
        case = parts[-1]   # 最后部分是大小写 (upper/lower)

        # 创建类别: letter_case (如: a_upper, a_lower)
        category = f"{letter}_{case}"

        if category not in bin_files:
            bin_files[category] = set()

        bin_files[category].add(filename_no_ext)

    # 打印统计信息
    for category, files in sorted(bin_files.items()):
        print(f"  {category}: {len(files)} bin文件")

    return bin_files


def scan_image_files(image_dir: str, required_views: List[str] = ["front_view.png", "side_view.png", "top_view.png"]) -> Dict[str, Set[str]]:
    """
    扫描image目录，找到有完整三视图的模型
    从文件夹名提取字母和大小写信息作为类别: letter_upper / letter_lower

    Returns:
        Dict[category, Set[model_name]]
    """
    image_path = Path(image_dir)
    image_models = {}

    print(f"扫描image目录: {image_dir}")

    for model_dir in image_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # 解析文件夹名格式: letter_fontname_case
        parts = model_name.split('_')
        if len(parts) < 3:
            print(f"  跳过格式不正确的文件夹: {model_name}")
            continue

        letter = parts[0]  # 第一部分是字母
        case = parts[-1]   # 最后部分是大小写 (upper/lower)

        # 创建类别: letter_case (如: a_upper, a_lower)
        category = f"{letter}_{case}"

        if category not in image_models:
            image_models[category] = set()

        # 检查是否有完整的三视图
        has_all_views = True
        for view_file in required_views:
            if not (model_dir / view_file).exists():
                has_all_views = False
                break

        if has_all_views:
            image_models[category].add(model_name)

    # 打印统计信息
    for category, models in sorted(image_models.items()):
        print(f"  {category}: {len(models)} 完整三视图模型")

    return image_models


def find_common_models(bin_files: Dict[str, Set[str]], image_models: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    找到既有bin文件又有三视图的模型

    Returns:
        Dict[category, Set[common_model_names]]
    """
    common_models = {}
    total_common = 0

    print("\n查找既有bin又有三视图的模型:")

    # 获取所有类别的并集
    all_categories = set(bin_files.keys()) | set(image_models.keys())

    for category in all_categories:
        bin_set = bin_files.get(category, set())
        image_set = image_models.get(category, set())

        # 找交集
        common_set = bin_set & image_set

        if common_set:
            common_models[category] = common_set
            total_common += len(common_set)
            print(f"  {category}: {len(common_set)} 个完整模型")
        else:
            if bin_set or image_set:
                print(f"  {category}: 0 个完整模型 (bin:{len(bin_set)}, image:{len(image_set)})")

    print(f"\n总计: {total_common} 个完整模型分布在 {len(common_models)} 个类别中")
    return common_models


def generate_annotations(common_models: Dict[str, Set[str]], bin_dir: str, image_dir: str, output_file: str, max_samples_per_category: int = None):
    """
    生成annotation JSON文件，使用绝对路径
    """
    annotations = {}
    bin_path = Path(bin_dir).resolve()
    image_path = Path(image_dir).resolve()

    # 添加类别信息
    categories = sorted(common_models.keys())
    annotations["_categories_info"] = {
        "available_categories": categories,
        "total_categories": len(categories),
        "note": f"Letters dataset with {len(categories)} categories and {sum(len(models) for models in common_models.values())} models"
    }

    # 为每个模型生成annotation
    for category in sorted(categories):
        models = common_models[category]

        # 如果指定了每个类别的最大样本数，则限制模型数量
        if max_samples_per_category is not None and len(models) > max_samples_per_category:
            models = sorted(models)[:max_samples_per_category]
            print(f"  限制类别 {category} 的样本数为 {max_samples_per_category} (原有 {len(common_models[category])} 个)")
        else:
            models = sorted(models)

        for model_name in models:
            # 生成唯一ID: category_modelname
            item_id = f"{category}_{model_name}"

            # 构建绝对路径 (Letters数据集是平铺结构，没有category子目录)
            bin_file_path = bin_path / f"{model_name}.bin"
            image_model_dir = image_path / model_name

            annotations[item_id] = {
                "metadata": {
                    "category": category,
                    "original_filename": model_name
                },
                "available_data": {
                    "views": {
                        "front": str(image_model_dir / "front_view.png"),
                        "side": str(image_model_dir / "side_view.png"),
                        "top": str(image_model_dir / "top_view.png")
                    },
                    "brep": str(bin_file_path)
                }
            }

    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    total_models = len(annotations) - 1  # 减去_categories_info
    print(f"\n[SUCCESS] 生成完成!")
    print(f"   输出文件: {output_file}")
    print(f"   类别数量: {len(categories)}")
    print(f"   模型数量: {total_models}")

    if max_samples_per_category is not None:
        print(f"   每类最大样本数: {max_samples_per_category}")

    # 统计实际生成的模型数量
    actual_counts = {}
    for item_id, item_data in annotations.items():
        if item_id != "_categories_info":
            category = item_data["metadata"]["category"]
            actual_counts[category] = actual_counts.get(category, 0) + 1

    # 打印每个类别的统计
    print(f"\n[STATS] 各类别统计:")
    for category in categories:
        original_count = len(common_models[category])
        actual_count = actual_counts.get(category, 0)
        if max_samples_per_category is not None and original_count > max_samples_per_category:
            print(f"   {category}: {actual_count} 个模型 (原有 {original_count} 个，已限制)")
        else:
            print(f"   {category}: {actual_count} 个模型")


def main():
    parser = argparse.ArgumentParser(description="生成Letters数据集annotation文件")
    parser.add_argument(
        "--bin_dir",
        type=str,
        default=r"/root/autodl-tmp/graph_with_eattr",
        help="bin文件目录路径"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=r"/root/autodl-tmp/image",
        help="image文件目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/cadnet/Network/letters_annotations.json",
        help="输出annotation文件路径"
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["front_view.png", "side_view.png", "top_view.png"],
        help="需要的视图文件名"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个类别的最大样本数量 (默认无限制)"
    )

    args = parser.parse_args()

    print("Letters数据集annotation生成器")
    print("=" * 50)

    # 检查目录是否存在
    if not Path(args.bin_dir).exists():
        print(f"[ERROR] bin目录不存在: {args.bin_dir}")
        return

    if not Path(args.image_dir).exists():
        print(f"[ERROR] image目录不存在: {args.image_dir}")
        return

    try:
        # 1. 扫描bin文件
        bin_files = scan_bin_files(args.bin_dir)

        # 2. 扫描image文件
        image_models = scan_image_files(args.image_dir, args.views)

        # 3. 找到共同模型
        common_models = find_common_models(bin_files, image_models)

        if not common_models:
            print("[ERROR] 没有找到既有bin又有完整三视图的模型!")
            return

        # 4. 生成annotation文件
        if args.max_samples:
            print(f"\n[INFO] 限制每个类别最多 {args.max_samples} 个样本")

        generate_annotations(common_models, args.bin_dir, args.image_dir, args.output, args.max_samples)

    except Exception as e:
        print(f"[ERROR] 生成过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()