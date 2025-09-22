#!/usr/bin/env python3
"""
测试图像编码器的示例脚本
展示如何使用多视图ViT编码器处理CAD零件图像
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.encoders.image_encoder import MultiViewViTEncoder, ImagePreprocessor
from src.data.transforms import create_train_transform, create_val_transform
from config.model_config import ModelConfig


def create_dummy_images(num_views: int = 3, size: tuple = (224, 224)) -> list:
    """创建测试用的虚拟图像"""
    images = []
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # 红、绿、蓝

    for i in range(num_views):
        # 创建带有不同图案的彩色图像
        img_array = np.ones((*size, 3), dtype=np.uint8) * 255
        color = colors[i % len(colors)]

        # 添加一些几何图案来模拟CAD零件
        center_x, center_y = size[0] // 2, size[1] // 2

        # 绘制圆形
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (size[0] // 4) ** 2
        img_array[mask] = color

        # 绘制矩形
        rect_size = size[0] // 6
        rect_x1, rect_y1 = center_x - rect_size, center_y - rect_size
        rect_x2, rect_y2 = center_x + rect_size, center_y + rect_size
        img_array[rect_y1:rect_y2, rect_x1:rect_x2] = [255 - c for c in color]

        images.append(Image.fromarray(img_array))

    return images


def test_image_encoder():
    """测试图像编码器功能"""
    print("=" * 60)
    print("测试多视图ViT图像编码器")
    print("=" * 60)

    # 配置
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建图像编码器
    print("\n1. 创建图像编码器...")
    try:
        encoder = MultiViewViTEncoder(
            model_name="vit_base_patch16_224",
            num_views=3,
            feature_dim=768,
            fusion_method="attention",
        ).to(device)
    except Exception as e:
        print(f"在线模式失败: {e}")
        print("切换到离线模式...")
        encoder = MultiViewViTEncoder(
            model_name="vit_base_patch16_224",
            num_views=3,
            feature_dim=768,
            fusion_method="attention",
        ).to(device)

    print(f"   - 模型参数数量: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"   - 可训练参数: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")

    # 创建图像预处理器
    print("\n2. 创建图像预处理器...")
    preprocessor = ImagePreprocessor(
        image_size=(224, 224),
        augment=False
    )

    # 创建测试数据
    print("\n3. 生成测试图像...")
    dummy_images = create_dummy_images(num_views=3)
    print(f"   - 生成了 {len(dummy_images)} 个视图的图像")

    # 预处理图像
    print("\n4. 预处理图像...")
    processed_images = preprocessor(dummy_images)  # [3, 3, 224, 224]
    batch_images = processed_images.unsqueeze(0).to(device)  # [1, 3, 3, 224, 224]
    print(f"   - 预处理后形状: {batch_images.shape}")

    # 前向传播
    print("\n5. 进行特征提取...")
    encoder.eval()
    with torch.no_grad():
        outputs = encoder(batch_images)

    # 输出结果
    print("\n6. 输出结果:")
    print(f"   - 融合特征形状: {outputs['fused_features'].shape}")
    print(f"   - 视图特征形状: {outputs['view_features'].shape}")
    print(f"   - 前视图特征形状: {outputs['individual_features']['front_view'].shape}")
    print(f"   - 侧视图特征形状: {outputs['individual_features']['side_view'].shape}")
    print(f"   - 俯视图特征形状: {outputs['individual_features']['top_view'].shape}")

    # 特征统计
    fused_features = outputs['fused_features']
    print(f"\n7. 特征统计:")
    print(f"   - 融合特征均值: {fused_features.mean().item():.4f}")
    print(f"   - 融合特征标准差: {fused_features.std().item():.4f}")
    print(f"   - 融合特征范围: [{fused_features.min().item():.4f}, {fused_features.max().item():.4f}]")

    return outputs


def test_different_fusion_methods():
    """测试不同的融合方法"""
    print("\n" + "=" * 60)
    print("测试不同的特征融合方法")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_methods = ["attention", "average", "max", "concat"]

    # 创建测试数据
    dummy_images = create_dummy_images(num_views=3)
    preprocessor = ImagePreprocessor(image_size=(224, 224), augment=False)
    processed_images = preprocessor(dummy_images)
    batch_images = processed_images.unsqueeze(0).to(device)

    results = {}

    for method in fusion_methods:
        print(f"\n测试融合方法: {method}")

        # 创建编码器
        try:
            encoder = MultiViewViTEncoder(
                model_name="vit_base_patch16_224",
                num_views=3,
                feature_dim=768,
                fusion_method=method,
                
            ).to(device)
        except Exception:
            # 如果在线失败，使用离线模式
            encoder = MultiViewViTEncoder(
                model_name="vit_base_patch16_224",

                num_views=3,
                feature_dim=768,
                fusion_method=method,
            ).to(device)

        # 前向传播
        encoder.eval()
        with torch.no_grad():
            outputs = encoder(batch_images)

        fused_features = outputs['fused_features']
        results[method] = {
            "shape": fused_features.shape,
            "mean": fused_features.mean().item(),
            "std": fused_features.std().item()
        }

        print(f"   - 输出形状: {fused_features.shape}")
        print(f"   - 特征均值: {fused_features.mean().item():.4f}")
        print(f"   - 特征标准差: {fused_features.std().item():.4f}")

    return results


def test_transforms():
    """测试图像变换功能"""
    print("\n" + "=" * 60)
    print("测试图像变换功能")
    print("=" * 60)

    # 创建测试图像
    dummy_images = create_dummy_images(num_views=3)

    # 测试训练变换
    print("\n1. 测试训练时变换（带增强）...")
    train_transform = create_train_transform(augment=True)
    train_transformed = train_transform(dummy_images)
    print(f"   - 变换后形状: {train_transformed.shape}")
    print(f"   - 数据范围: [{train_transformed.min():.3f}, {train_transformed.max():.3f}]")

    # 测试验证变换
    print("\n2. 测试验证时变换（无增强）...")
    val_transform = create_val_transform()
    val_transformed = val_transform(dummy_images)
    print(f"   - 变换后形状: {val_transformed.shape}")
    print(f"   - 数据范围: [{val_transformed.min():.3f}, {val_transformed.max():.3f}]")

    return train_transformed, val_transformed


def visualize_features(outputs):
    """可视化特征分布"""
    print("\n" + "=" * 60)
    print("可视化特征分布")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 融合特征分布
        fused_features = outputs['fused_features'].cpu().numpy().flatten()
        axes[0, 0].hist(fused_features, bins=50, alpha=0.7)
        axes[0, 0].set_title("融合特征分布")
        axes[0, 0].set_xlabel("特征值")
        axes[0, 0].set_ylabel("频次")

        # 各视图特征均值对比
        view_features = outputs['view_features'].cpu().numpy()  # [1, 3, 768]
        view_means = view_features.mean(axis=2).flatten()  # [3]
        view_names = ["前视图", "侧视图", "俯视图"]

        axes[0, 1].bar(view_names, view_means)
        axes[0, 1].set_title("各视图特征均值")
        axes[0, 1].set_ylabel("特征均值")

        # 特征相关性热图
        individual_features = torch.stack([
            outputs['individual_features']['front_view'],
            outputs['individual_features']['side_view'],
            outputs['individual_features']['top_view']
        ]).cpu().numpy()  # [3, 1, 768]

        correlation_matrix = np.corrcoef(individual_features.reshape(3, -1))
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title("视图间特征相关性")
        axes[1, 0].set_xticks(range(3))
        axes[1, 0].set_yticks(range(3))
        axes[1, 0].set_xticklabels(view_names)
        axes[1, 0].set_yticklabels(view_names)
        plt.colorbar(im, ax=axes[1, 0])

        # 特征维度分析
        feature_std = view_features.std(axis=0).flatten()  # [768]
        axes[1, 1].plot(feature_std)
        axes[1, 1].set_title("特征维度标准差")
        axes[1, 1].set_xlabel("特征维度")
        axes[1, 1].set_ylabel("标准差")

        plt.tight_layout()
        plt.savefig("feature_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("特征分析图已保存为 feature_analysis.png")

    except ImportError:
        print("matplotlib 未安装，跳过可视化")
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")


def main():
    """主函数"""
    print("CAD图像编码器测试脚本")
    print("=" * 60)

    try:
        # 基础功能测试
        outputs = test_image_encoder()

        # 融合方法对比测试
        fusion_results = test_different_fusion_methods()

        # 变换功能测试
        transform_results = test_transforms()

        # 特征可视化
        visualize_features(outputs)

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

        # 输出融合方法对比结果
        print("\n融合方法对比结果:")
        for method, result in fusion_results.items():
            print(f"  {method:>10}: 形状={result['shape']}, 均值={result['mean']:.4f}, 标准差={result['std']:.4f}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()