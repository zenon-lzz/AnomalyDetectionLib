"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Data Factory Module for Time Series Datasets
    This module provides a unified interface for loading different time series datasets
    for anomaly detection. 
==================================================
"""
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

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


def data_provider(args: ConfigType, split_way: str = 'train_no_split', validate_proportion=0.2) -> tuple[
    DataLoader[Any], DataLoader[Any] | None, DataLoader[Any]]:
    dataset_class = dataset_dict[args.dataset]
    batch_size = args.batch_size

    test_dataset = dataset_class(root_path=args.dataset_root_path, win_size=args.window_size, mode='test')

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    train_dataset = dataset_class(root_path=args.dataset_root_path, win_size=args.window_size, mode='train')

    train_length = len(train_dataset)

    if split_way == 'train_split_proportionally':
        indices = torch.arange(train_length)
        validate_start_index = int(train_length * (1 - validate_proportion))
        train_subset = Subset(train_dataset, indices[:validate_start_index])
        validate_subset = Subset(train_subset, indices[validate_start_index:])
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=True)
        validate_dataloader = DataLoader(validate_subset, batch_size=batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
        return train_dataloader, validate_dataloader, test_dataloader
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=True)
        return train_dataloader, None, test_dataloader
