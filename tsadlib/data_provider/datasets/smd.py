"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: SMD (Server Machine Dataset) Dataset
    This module implements a PyTorch Dataset for the Server Machine Dataset,
    which contains server performance metrics for anomaly detection.
==================================================
"""
import os

import numpy as np

from .base import BaseDataset
from ... import PreprocessScalerEnum, ConfigType


class SMDDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Server Machine Dataset (SMD).
    
    This dataset contains multivariate time series data collected from server machines,
    including CPU usage, memory utilization, network traffic, and other performance metrics.
    The dataset is designed for anomaly detection in server monitoring applications.

    """

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        """
        Initialize the SMD dataset.

        """
        # Initialize the base class with window parameters
        super().__init__(args.window_size, args.window_stride, mode)

        # Initialize and fit StandardScaler on training data
        self.set_scaler(scaler)
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':
            # Store normalized training data
            self.train = data
        elif mode == 'test':
            # Load and transform test data
            test_data = np.load(os.path.join(root_path, 'SMD_test.npy'))
            self.test = self.scaler.transform(test_data)
            # Load test labels
            self.test_labels = np.load(os.path.join(root_path, 'SMD_test_label.npy'))
