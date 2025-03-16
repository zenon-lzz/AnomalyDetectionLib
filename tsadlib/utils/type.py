"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-15
@Description：模型配置参数约束
==================================================
"""
from dataclasses import dataclass


@dataclass
class ConfigType:
    seq_len: int
    top_k: int
    d_model: int
    d_ff: int
    num_kernels: int
    e_layers: int
    enc_in: int
    c_out: int
    dropout: float
    embed_type: str
    freq: str
