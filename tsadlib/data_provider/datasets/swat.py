"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: SWaT (Secure Water Treatment) Dataset
    This module implements a PyTorch Dataset for the SWaT dataset,
    which contains operational data from a real-world water treatment facility.
    The dataset includes normal operations and cyber-attacks for anomaly detection research.
==================================================
"""
import os

import pandas as pd

from .base import BaseDataset
from ... import PreprocessScalerEnum, ConfigType


class SWaTDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Secure Water Treatment (SWaT) dataset.
    
    This dataset contains multivariate time series data collected from a scaled-down
    water treatment testbed. It includes sensor measurements and actuator states
    during normal operation and cyber-attack scenarios.

    """

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        """
        Initialize the SWaT dataset.
        

        """
        # Initialize the base class with window parameters
        super().__init__(args.window_size, args.window_stride, mode)

        # Load training data and exclude the label column
        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv')).values[:, :-1]
        self.set_scaler(scaler)
        self.scaler.fit(train_data)
        data = self.scaler.transform(train_data)

        if mode == 'train':
            # Store normalized training data
            self.train = data
        elif mode == 'test':
            # Load test data and separate features from labels
            test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
            # Transform features (excluding label column)
            self.test = self.scaler.transform(test_data.values[:, :-1])
            # Extract labels (last column)
            self.test_labels = test_data.values[:, -1:]
