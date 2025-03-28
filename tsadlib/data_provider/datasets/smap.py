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
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class SMAPDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Soil Moisture Active Passive (SMAP) anomaly detection dataset.
    
    This dataset contains telemetry data from NASA's SMAP satellite mission.
    The class handles data loading, preprocessing, and windowing for time series analysis.
    It inherits window handling functionality from the BaseDataset class.
    
    Attributes:
        scaler (StandardScaler): Scaler for data normalization
        train (np.ndarray): Training data after normalization
        test (np.ndarray): Test data after normalization
        test_labels (np.ndarray): Anomaly labels for test data
    """

    def __init__(self, root_path, win_size, step=1, mode="train"):
        """
        Initialize the SMAP dataset.
        
        Args:
            root_path (str): Path to the dataset files
            win_size (int): Size of the sliding window
            step (int): Step size for the sliding window
            mode (str): 'train' or 'test' mode
        """
        # Initialize the base class with window parameters
        super(SMAPDataset, self).__init__(win_size, step, mode)
        
        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
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
