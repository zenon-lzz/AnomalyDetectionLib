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
    root_path: str = field()  # Root path to dataset

    # Model Architecture Parameters
    top_k: int = field()  # Top k time-frequency combinations (TimesNet)
    dimension_model: int = field()  # Model dimension
    dimension_fcl: int = field()  # Feed-forward layer dimension
    num_kernels: int = field()  # Number of convolutional kernels
    encoder_layers: int = field()  # Number of encoder layers
    input_channels: int = field()  # Input channel dimension
    output_channels: int = field()  # Output channel dimension
    dropout: float = field()  # Dropout rate for regularization
    batch_size: int = field()  # Batch size for training/testing
    window_length: int = field()  # Sequence/window length for time series

    # Training Parameters
    learning_rate: float = field()  # Learning rate for optimizer
    anomaly_ratio: float = field()  # Expected ratio of anomalies in data

    # Parameters that have default value
    embedding_type: str = field(default='timeF')  # Type of embedding (e.g., 'timeF')
    freq: str = field(default='h')  # freq for time features encoding,
    # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
    # you can also use more detailed freq like 15min or 3h

    # Training Parameters
    train_epochs: int = field(default=10)  # Number of training epochs
    patience: int = field(default=10)  # Patience for early stopping
    num_epochs: int = field(default=20)  # Alternative epoch specification

    # Autoconfigure DataLoader's num_workers based on debug status and system
    num_workers: int = field(
        default=0 if IS_DEBUG else min(10, multiprocessing.cpu_count())
    )  # 0 in debug mode, otherwise use CPU count with max of 10
