"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Data Factory Module for Time Series Datasets
    This module provides a unified interface for loading different time series datasets
    for anomaly detection. 
==================================================
"""
import os.path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from tsadlib.configs.type import ConfigType, DatasetSplitEnum
from .datasets.msl import MSLDataset
from .datasets.nips_ts import NIPSTSWaterDataset, NIPSTSSwanDataset, NIPSTSCreditcardDataset
from .datasets.psm import PSMDataset
from .datasets.smap import SMAPDataset
from .datasets.smd import SMDDataset
from .datasets.swat import SWaTDataset
from .datasets.wadi import WADIDataset

# Registry of available datasets
# Maps dataset names to their corresponding Dataset classes
dataset_dict = {
    'MSL': MSLDataset,
    'SMAP': SMAPDataset,
    'SMD': SMDDataset,
    'SWaT': SWaTDataset,
    'PSM': PSMDataset,
    'WADI': WADIDataset,
    'NIPS_TS_Water': NIPSTSWaterDataset,
    'NIPS_TS_Swan': NIPSTSSwanDataset,
    'NIPS_TS_Creditcard': NIPSTSCreditcardDataset
}


def data_provider(args: ConfigType, split_way: DatasetSplitEnum = DatasetSplitEnum.TRAIN_NO_SPLIT,
                  validate_proportion=0.2, k_proportion=0.1) -> \
        tuple[DataLoader[Any] | None, ...]:
    """
    Factory function that provides data loaders for time series anomaly detection datasets.
    
    This function creates and configures DataLoader objects for training, validation, and testing
    based on the specified dataset and splitting strategy.
    
    Args:
        args (ConfigType): Configuration parameters containing dataset, paths, and loader settings
        split_way (str): Strategy for splitting the dataset:
            - 'TRAIN_NO_SPLIT': No validation set, only train and test
            - 'TRAIN_VALIDATE_SPLIT_WITH_DUPLICATES': Select the data with a final proportion of 'validate_proportion' in the training set as the validation set
            - 'TRAIN_VALIDATE_SPLIT': Split training data into train and validation sets
            - 'TRAIN_VALIDATE_K_SPLIT': Split into train, validation, and k-subset (for few-shot learning)
        validate_proportion (float): Proportion of training data to use for validation (default: 0.2)
        k_proportion (float): Proportion of data to use for k-subset in few-shot learning (default: 0.1)
        
    Returns:
        tuple: Combination of DataLoaders depending on split_way:
            - (train_loader, None, test_loader) for 'train_no_split'
            - (train_loader, validate_loader, test_loader) for 'train_validate_split'
            - (train_loader, validate_loader, test_loader, k_loader) for 'train_validate_k_split'
    """
    # Get the appropriate dataset class based on the dataset name in args
    dataset_class = dataset_dict[args.dataset]
    batch_size = args.batch_size

    root_path = os.path.join(args.dataset_root_path, args.dataset)

    # Create test dataset and dataloader
    test_dataset = dataset_class(root_path=root_path, win_size=args.window_size, step=args.window_size,
                                 mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    # Create training dataset
    train_dataset = dataset_class(root_path=root_path, win_size=args.window_size, step=args.window_step,
                                  mode='train')
    train_length = len(train_dataset)

    if split_way == DatasetSplitEnum.TRAIN_VALIDATE_SPLIT_WITH_DUPLICATES:
        # Select the data with a final proportion of 'validate_proportion' in the training set as the validation set
        indices = torch.arange(train_length)
        validate_start_index = int(train_length * (1 - validate_proportion))

        # Create subsets for training and validation
        validate_subset = Subset(train_dataset, indices[validate_start_index:])

        # Create data loaders for training and validation
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        validate_dataloader = DataLoader(validate_subset, batch_size=batch_size, num_workers=args.num_workers)

        return train_dataloader, validate_dataloader, test_dataloader
    elif split_way == DatasetSplitEnum.TRAIN_VALIDATE_SPLIT:
        # Split training data into training and validation sets
        indices = torch.arange(train_length)
        validate_start_index = int(train_length * (1 - validate_proportion))

        # Create subsets for training and validation
        train_subset = Subset(train_dataset, indices[:validate_start_index])
        validate_subset = Subset(train_dataset, indices[validate_start_index:])

        # Create data loaders for training and validation
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=args.window_step)
        validate_dataloader = DataLoader(validate_subset, batch_size=batch_size, num_workers=args.num_workers)

        return train_dataloader, validate_dataloader, test_dataloader

    elif split_way == DatasetSplitEnum.TRAIN_VALIDATE_K_SPLIT:
        # Split for few-shot learning scenario with train, validation, and k-subset
        indices = torch.arange(train_length)
        validate_start_index = int(train_length * (1 - validate_proportion))

        # Create subsets for training and validation
        train_subset = Subset(train_dataset, indices[:validate_start_index])
        validate_subset = Subset(train_dataset, indices[validate_start_index:])

        # Create k-subset for few-shot learning (small portion of training data)
        k_end_index = int(train_length * k_proportion)
        k_subset = Subset(train_dataset, indices[:k_end_index])

        # Create data loaders for all three subsets
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=True)
        validate_dataloader = DataLoader(validate_subset, batch_size=batch_size, num_workers=args.num_workers)
        k_dataloader = DataLoader(k_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)

        return train_dataloader, validate_dataloader, test_dataloader, k_dataloader

    else:
        # Default case: no validation split, just training and testing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=True)

        return train_dataloader, test_dataloader
