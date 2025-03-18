"""
=================================================
@Author: Zenon
@Date: 2025-03-15
@Description：模型配置参数约束
==================================================
"""
from dataclasses import dataclass, field


@dataclass
class ConfigType:
    model: str = field(default='TimesNet')
    dataset: str = field(default=None)
    root_path: str = field(default=None)

    batch_size: int = field(default=None)
    seq_len: int = field(default=None)
    num_workers: int = field(default=None)

    top_k: int = field(default=None)
    d_model: int = field(default=None)
    d_ff: int = field(default=None)
    num_kernels: int = field(default=None)
    e_layers: int = field(default=None)
    enc_in: int = field(default=None)
    c_out: int = field(default=None)
    dropout: float = field(default=None)
    embed_type: str = field(default=None)
    freq: str = field(default=None)

    learning_rate: float = field(default=None)
    anomaly_ratio: float = field(default=None)
    train_epochs: int = field(default=10)
    patience: int = field(default=3)
    num_epochs: int = field(default=3)
