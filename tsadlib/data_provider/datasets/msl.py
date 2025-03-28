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
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class MSLDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Mars Science Laboratory (MSL) anomaly detection dataset.
    
    This dataset contains telemetry data from the Mars Science Laboratory rover mission.
    The class handles data loading, preprocessing, and windowing for time series analysis.
    It inherits window handling functionality from the BaseDataset class.
    
    Attributes:
        scaler (StandardScaler): Scaler for data normalization
        train (np.ndarray): Training data after normalization
        test (np.ndarray): Test data after normalization
        test_labels (np.ndarray): Anomaly labels for test data
    """

    def __init__(self, root_path, win_size, step=1, mode='train'):
        """
        Initialize the MSL dataset.
        
        Args:
            root_path (str): Path to the dataset files
            win_size (int): Size of the sliding window
            step (int): Step size for the sliding window
            mode (str): 'train' or 'test' mode
        """
        # Initialize the base class with window parameters
        super(MSLDataset, self).__init__(win_size, step, mode)

        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
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
