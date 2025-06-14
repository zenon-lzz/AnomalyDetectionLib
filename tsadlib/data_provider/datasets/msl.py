"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: MSL (Mars Science Laboratory) Dataset
    This module provides a PyTorch Dataset implementation for the MSL anomaly detection dataset.
    The MSL dataset contains telemetry data from the Mars Science Laboratory rover mission,
    with labeled anomalies for testing and evaluation of anomaly detection algorithms.
==================================================
"""
import os

import numpy as np

from .base import BaseDataset
from ... import ConfigType, PreprocessScalerEnum


class MSLDataset(BaseDataset):

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        # Initialize the base class with window parameters
        super().__init__(args.window_size, args.window_stride, mode)

        # Initialize and fit StandardScaler on training data
        self.set_scaler(scaler)
        data = np.load(os.path.join(root_path, 'MSL_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':
            # Store normalized training data
            self.train = data
        elif mode == 'test':
            # Load and transform test data
            test_data = np.load(os.path.join(root_path, 'MSL_test.npy'))
            self.test = self.scaler.transform(test_data)
            # Load test labels
            self.test_labels = np.load(os.path.join(root_path, 'MSL_test_label.npy'))
