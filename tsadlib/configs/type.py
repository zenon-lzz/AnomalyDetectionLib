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
from dataclasses import dataclass, field
from enum import Enum



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

    # data preprocessing parameters

    # Model Architecture Parameters
    d_model: int = field()  # Model dimension
    input_channels: int = field()  # Input channel dimension
    output_channels: int = field()  # Output channel dimension
    dropout: float = field()  # Dropout rate for regularization
    batch_size: int = field()  # Batch size for training/testing
    window_size: int = field()  # Sequence/window length for time series
    window_stride: int = field()  # Sliding window stride

    # Parameters that have default value
    task_name: str = field(default='benchmarks')
    checkpoints: str = field(default='checkpoints')
    mode: str = field(default='train')  # Model's execution status, options: ['train', 'test']
    use_gpu: bool = field(default=True)
    gpu_type: str = field(default='cuda')
    use_multi_gpu: str = field(default=False)
    gpu: int = field(default=0)
    devices: str = field(default='')


    # Model Architecture Parameters
    dimension_fcl: int = field(default=16)  # Feed-forward layer dimension
    temporal_dimension_fcl: int = field(default=16)  # temporal transformer fcl layer layer dimension
    spatio_dimension_fcl: int = field(default=16)  # spatio transformer fcl layer layer dimension
    encoder_layers: int = field(default=1)  # Number of encoder layers
    temporal_encoder_layers: int = field(default=1)  # Number of temporal encoder layers
    spatio_encoder_layers: int = field(default=1)  # Number of spatial encoder layers
    top_k: int = field(default=3)  # Top k time-frequency combinations (TimesNet)
    dataset: str = field(default='MSL')  # Dataset name (e.g., 'MSL', 'SMAP', 'SMD')
    dataset_root_path: str = field(default='data')  # Root path to dataset
    num_kernels: int = field(default=6)  # Number of convolutional kernels
    hyper_parameter_lambda: float = field(default=0.01)  # Loss term coefficient
    n_heads: int = field(default=8)  # The number of heads in Multiple Head Attention
    num_memory: int = field(default=10)  # The number of Memory slots
    temperature: float = field(default=0.1)  # The latent space deviation hyperparameter
    patch_list: list[int] = field(default_factory=lambda: [10, 20])
    kernel_list: list[int] = field(default_factory=lambda: [5])

    # Training Parameters
    num_epochs: int = field(default=10)  # Number of training epochs
    runs: int = field(default=1)
    patience: int = field(default=10)  # Patience for early stopping
    learning_rate: float = field(default=1e-4)  # Learning rate for optimizer
    end_learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=5e-5)
    warmup_epoch: int = field(default=0)
    anomaly_ratio: float = field(default=1)  # Expected ratio of anomalies in data

    use_tensorboard: bool = field(default=False)
    use_wandb: bool = field(default=False)


@dataclass
class Metric:
    """
    Data class for storing anomaly detection metrics results.
    Supports partial initialization with default values.
    """
    # Common metrics
    Precision: float = field(default=0.0)
    Recall: float = field(default=0.0)
    F1_score: float = field(default=0.0)
    ROC_AUC: float = field(default=0.0)

    # Enhanced metrics
    Affiliation_Precision: float = field(default=0.0)
    Affiliation_Recall: float = field(default=0.0)
    R_AUC_ROC: float = field(default=0.0)
    R_AUC_PR: float = field(default=0.0)
    VUS_ROC: float = field(default=0.0)
    VUS_PR: float = field(default=0.0)


class DatasetSplitEnum(Enum):
    TRAIN_NO_SPLIT = 'train_no_split'
    TRAIN_VALIDATE_SPLIT = 'train_validate_split'
    TRAIN_VALIDATE_SPLIT_WITH_DUPLICATES = 'train_validate_split_with_duplicates'
    TRAIN_VALIDATE_K_SPLIT = 'train_validate_k_split'


class EarlyStoppingModeEnum(Enum):
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class ValidateMetricEnum(Enum):
    LOSS = 'loss'
    F1_SCORE = 'f1_score'


class ThresholdWayEnum(Enum):
    BEST_F1 = 'best_f1'
    PERCENTILE = 'percentile'


class PreprocessScalerEnum(Enum):
    STANDARD = 'standard'
    MINMAX = 'min-max'
