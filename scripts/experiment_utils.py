#!/usr/bin/env python3
"""
实验管理工具
提供检查点管理、TensorBoard启动等实用功能
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.base_config import BaseConfig


def list_experiments(experiments_dir: str = None) -> List[Dict]:
    """列出所有实验"""
    if experiments_dir is None:
        config = BaseConfig()
        experiments_dir = config.model_save_dir

    experiments = []

    if not os.path.exists(experiments_dir):
        return experiments

    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        if os.path.isdir(exp_path):
            exp_info = {
                'name': exp_name,
                'path': exp_path,
                'created_time': datetime.fromtimestamp(os.path.getctime(exp_path)),
                'checkpoints': [],
                'best_models': []
            }

            # 查找检查点和最佳模型
            for file in os.listdir(exp_path):
                if file.endswith('.pth'):
                    file_path = os.path.join(exp_path, file)
                    if file.startswith('checkpoint_'):
                        exp_info['checkpoints'].append({
                            'name': file,
                            'path': file_path,
                            'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path))
                        })
                    elif file.startswith('best_model_'):
                        exp_info['best_models'].append({
                            'name': file,
                            'path': file_path,
                            'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path))
                        })

            # 按时间排序
            exp_info['checkpoints'].sort(key=lambda x: x['modified_time'], reverse=True)
            exp_info['best_models'].sort(key=lambda x: x['modified_time'], reverse=True)

            experiments.append(exp_info)

    # 按创建时间排序
    experiments.sort(key=lambda x: x['created_time'], reverse=True)
    return experiments


def get_checkpoint_info(checkpoint_path: str) -> Dict:
    """获取检查点信息"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        info = {
            'path': checkpoint_path,
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'best_accuracy': checkpoint.get('best_accuracy', 'Unknown'),
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
        }

        # 训练指标
        if 'train_metrics' in checkpoint:
            info['train_metrics'] = checkpoint['train_metrics']

        # 验证指标
        if 'val_metrics' in checkpoint:
            info['val_metrics'] = checkpoint['val_metrics']

        # 配置信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            if hasattr(config, '__dict__'):
                info['config'] = config.__dict__
            else:
                info['config'] = str(config)

        return info

    except Exception as e:
        return {
            'path': checkpoint_path,
            'error': str(e)
        }


def start_tensorboard(log_dir: str = None, port: int = 6006):
    """启动TensorBoard服务"""
    if log_dir is None:
        config = BaseConfig()
        log_dir = config.tensorboard_log_dir

    if not os.path.exists(log_dir):
        print(f"TensorBoard日志目录不存在: {log_dir}")
        return

    import subprocess
    try:
        cmd = f"tensorboard --logdir={log_dir} --port={port}"
        print(f"启动TensorBoard: {cmd}")
        print(f"请在浏览器中访问: http://localhost:{port}")
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        print("\nTensorBoard服务已停止")
    except Exception as e:
        print(f"启动TensorBoard失败: {e}")


def clean_old_checkpoints(experiment_name: str, keep_last_n: int = 5, experiments_dir: str = None):
    """清理旧的检查点文件"""
    if experiments_dir is None:
        config = BaseConfig()
        experiments_dir = config.model_save_dir

    exp_path = os.path.join(experiments_dir, experiment_name)
    if not os.path.exists(exp_path):
        print(f"实验目录不存在: {exp_path}")
        return

    # 获取所有检查点文件
    checkpoints = []
    for file in os.listdir(exp_path):
        if file.startswith('checkpoint_') and file.endswith('.pth'):
            file_path = os.path.join(exp_path, file)
            checkpoints.append({
                'name': file,
                'path': file_path,
                'modified_time': os.path.getmtime(file_path)
            })

    # 按修改时间排序
    checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)

    # 删除多余的检查点
    if len(checkpoints) > keep_last_n:
        to_delete = checkpoints[keep_last_n:]
        print(f"发现 {len(checkpoints)} 个检查点，保留最新的 {keep_last_n} 个")

        for checkpoint in to_delete:
            try:
                os.remove(checkpoint['path'])
                print(f"删除: {checkpoint['name']}")
            except Exception as e:
                print(f"删除失败 {checkpoint['name']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="实验管理工具")

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 列出实验
    list_parser = subparsers.add_parser('list', help='列出所有实验')
    list_parser.add_argument('--experiments_dir', type=str, help='实验目录路径')

    # 检查点信息
    info_parser = subparsers.add_parser('info', help='显示检查点信息')
    info_parser.add_argument('checkpoint_path', type=str, help='检查点文件路径')

    # 启动TensorBoard
    tb_parser = subparsers.add_parser('tensorboard', help='启动TensorBoard')
    tb_parser.add_argument('--log_dir', type=str, help='日志目录路径')
    tb_parser.add_argument('--port', type=int, default=6006, help='端口号')

    # 清理检查点
    clean_parser = subparsers.add_parser('clean', help='清理旧的检查点')
    clean_parser.add_argument('experiment_name', type=str, help='实验名称')
    clean_parser.add_argument('--keep', type=int, default=5, help='保留最新的N个检查点')
    clean_parser.add_argument('--experiments_dir', type=str, help='实验目录路径')

    args = parser.parse_args()

    if args.command == 'list':
        experiments = list_experiments(args.experiments_dir)

        if not experiments:
            print("未找到任何实验")
            return

        print(f"找到 {len(experiments)} 个实验:\n")

        for exp in experiments:
            print(f"实验名称: {exp['name']}")
            print(f"创建时间: {exp['created_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"检查点数量: {len(exp['checkpoints'])}")
            print(f"最佳模型数量: {len(exp['best_models'])}")

            if exp['checkpoints']:
                latest = exp['checkpoints'][0]
                print(f"最新检查点: {latest['name']} ({latest['modified_time'].strftime('%Y-%m-%d %H:%M:%S')})")

            if exp['best_models']:
                best = exp['best_models'][0]
                print(f"最佳模型: {best['name']} ({best['modified_time'].strftime('%Y-%m-%d %H:%M:%S')})")

            print("-" * 60)

    elif args.command == 'info':
        info = get_checkpoint_info(args.checkpoint_path)

        if 'error' in info:
            print(f"读取检查点失败: {info['error']}")
            return

        print(f"检查点信息: {info['path']}")
        print(f"Epoch: {info['epoch']}")
        print(f"最佳准确率: {info['best_accuracy']}")
        print(f"包含优化器状态: {info['has_optimizer']}")
        print(f"包含调度器状态: {info['has_scheduler']}")

        if 'train_metrics' in info:
            print(f"训练指标: {info['train_metrics']}")

        if 'val_metrics' in info:
            print(f"验证指标: {info['val_metrics']}")

    elif args.command == 'tensorboard':
        start_tensorboard(args.log_dir, args.port)

    elif args.command == 'clean':
        clean_old_checkpoints(args.experiment_name, args.keep, args.experiments_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()