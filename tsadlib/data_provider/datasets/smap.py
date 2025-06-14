"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: SMAP (Soil Moisture Active Passive) Dataset
    This module provides a PyTorch Dataset implementation for the SMAP anomaly detection dataset.
    The SMAP dataset contains telemetry data from NASA's Soil Moisture Active Passive satellite,
    with labeled anomalies for testing and evaluation of anomaly detection algorithms.
==================================================
"""
import os

import numpy as np

from .base import BaseDataset
from ... import PreprocessScalerEnum, ConfigType


class SMAPDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Soil Moisture Active Passive (SMAP) anomaly detection dataset.
    
    This dataset contains telemetry data from NASA's SMAP satellite mission.
    The class handles data loading, preprocessing, and windowing for time series analysis.
    It inherits window handling functionality from the BaseDataset class.

    """

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        """
        Initialize the SMAP dataset.
        

        """
        # Initialize the base class with window parameters
        super().__init__(args.window_size, args.window_stride, mode)
        
        # Initialize and fit StandardScaler on training data
        self.set_scaler(scaler)
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':
            # Store normalized training data
            self.train = data
        elif mode == 'test':
            # Load and transform test data
            test_data = np.load(os.path.join(root_path, 'SMAP_test.npy'))
            self.test = self.scaler.transform(test_data)
            # Load test labels
            self.test_labels = np.load(os.path.join(root_path, 'SMAP_test_label.npy'))
