import os
from dataclasses import dataclass
from typing import List


@dataclass
class BaseConfig:
    """基础配置类"""

    # 项目路径
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root: str = os.path.join(project_root, "data")
    model_save_dir: str = os.path.join(project_root, "experiments", "checkpoints")
    log_dir: str = os.path.join(project_root, "experiments", "logs")
    tensorboard_log_dir: str = os.path.join(project_root, "experiments", "tensorboard")

    # 设备配置
    device: str = "cuda"
    num_workers: int = 4

    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 随机种子
    seed: int = 42

    # 日志配置
    log_level: str = "INFO"
    save_interval: int = 10
    eval_interval: int = 5

    # 恢复训练配置
    resume_from_checkpoint: str = ""

    # TensorBoard配置
    log_histograms: bool = True
    log_gradients: bool = False
    histogram_freq: int = 10

    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)