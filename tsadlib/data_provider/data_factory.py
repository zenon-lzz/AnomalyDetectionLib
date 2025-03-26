"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Data Factory Module for Time Series Datasets
    This module provides a unified interface for loading different time series datasets
    for anomaly detection. 
==================================================
"""
from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from tsadlib.configs.type import ConfigType
from .datasets.msl import MSLDataset
from .datasets.smap import SMAPDataset
from .datasets.smd import SMDDataset
from .datasets.swat import SWATDataset

# Registry of available datasets
# Maps dataset names to their corresponding Dataset classes
dataset_dict = {
    'MSL': MSLDataset,
    'SMAP': SMAPDataset,
    'SMD': SMDDataset,
    'SWAT': SWATDataset
}


def data_provider(args: ConfigType, flag: str) -> Tuple[Dataset, DataLoader]:
    """
    Creates and configures dataset and dataloader for specified dataset type.
    
    Features:
    - Dynamic dataset selection
    - Automatic shuffle configuration
    - Consistent interface across datasets
    
    Args:
        args (ConfigType): Configuration containing:
            - dataset: Name of dataset to use
            - root_path: Path to dataset files
            - window_length: Sequence length for windows
            - batch_size: Size of batches
            - num_workers: Number of worker processes
        flag (str): Dataset split identifier ('train'/'val'/'test')
    
    Returns:
        Tuple[Dataset, DataLoader]: 
            - Configured dataset instance
            - DataLoader with appropriate settings
    
    Note:
        - Shuffling is disabled for test set
        - All datasets use the same window-based loading mechanism
        - Incomplete final batches are kept (drop_last=False)
    """
    # Select appropriate dataset class
    dataset_class = dataset_dict[args.dataset]
    drop_last = False
    batch_size = args.batch_size
    # Disable shuffling for test set to maintain temporal order
    shuffle_flag = False if flag == 'test' else True

    # Initialize dataset with common configuration
    dataset = dataset_class(
        args=args,
        root_path=args.root_path,
        win_size=args.window_length,
        flag=flag,
    )

    # Configure DataLoader with standard settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return dataset, dataloader
