"""
=================================================
@Author: Zenon
@Date: 2025-03-15
@Description: Model Configuration Parameter Constraints
    This module defines the configuration dataclass for time series
    anomaly detection models. It provides a structured way to specify
    model parameters, dataset information, and training settings.
==================================================
"""
import multiprocessing
from dataclasses import dataclass, field

from tsadlib.configs.constants import IS_DEBUG


@dataclass
class ConfigType:
    """
    Configuration class for time series anomaly detection models.
    
    This class serves as a central configuration hub for all models in the tsadlib
    package. It defines parameters for data loading, model architecture, and training
    settings. The class uses Python's dataclass for clean parameter definition.
    """
    # Parameters that must be provided

    model: str = field()  # Model name (e.g., 'TimesNet')

    # Dataset Information
    dataset: str = field()  # Dataset name (e.g., 'MSL', 'SMAP', 'SMD')
    dataset_root_path: str = field()  # Root path to dataset

    # Model Architecture Parameters
    top_k: int = field()  # Top k time-frequency combinations (TimesNet)
    d_model: int = field()  # Model dimension
    dimension_fcl: int = field()  # Feed-forward layer dimension
    num_kernels: int = field()  # Number of convolutional kernels
    encoder_layers: int = field()  # Number of encoder layers
    input_channels: int = field()  # Input channel dimension
    output_channels: int = field()  # Output channel dimension
    dropout: float = field()  # Dropout rate for regularization
    batch_size: int = field()  # Batch size for training/testing
    window_size: int = field()  # Sequence/window length for time series

    # Training Parameters
    learning_rate: float = field()  # Learning rate for optimizer
    anomaly_ratio: float = field()  # Expected ratio of anomalies in data

    # Parameters that have default value
    mode: str = field(default='train')  # Model's execution status, options: ['train', 'test']
    # Model Architecture Parameters
    embedding_type: str = field(default='normal')  # Type of embedding (e.g., 'timeF')
    freq: str = field(default='h')  # freq for time features encoding,
    # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
    # you can also use more detailed freq like 15min or 3h
    # False: using random initialization, True: using K-Means
    hyper_parameter_lambda: float = field(default=0.01)
    n_heads: int = field(default=8)  # The number of heads in Multiple Head Attention
    num_memory: int = field(default=10)  # The number of Memory slots
    temperature: float = field(default=0.1)  # The latent space deviation hyperparameter in MEMTO

    # Training Parameters
    num_epochs: int = field(default=10)  # Number of training epochs
    patience: int = field(default=10)  # Patience for early stopping

    # Autoconfigure DataLoader's num_workers based on debug status and system
    num_workers: int = field(
        default=0 if IS_DEBUG else min(10, multiprocessing.cpu_count())
    )  # 0 in debug mode, otherwise use CPU count with max of 10
