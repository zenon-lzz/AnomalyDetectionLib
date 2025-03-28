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
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class SMDDataset(BaseDataset):
    """
    PyTorch Dataset implementation for the Server Machine Dataset (SMD).
    
    This dataset contains multivariate time series data collected from server machines,
    including CPU usage, memory utilization, network traffic, and other performance metrics.
    The dataset is designed for anomaly detection in server monitoring applications.
    
    Attributes:
        scaler (StandardScaler): Scaler for data normalization
        train (np.ndarray): Training data after normalization
        test (np.ndarray): Test data after normalization
        test_labels (np.ndarray): Anomaly labels for test data
    """

    def __init__(self, root_path, win_size, step=100, mode="train"):
        """
        Initialize the SMD dataset.
        
        Args:
            root_path (str): Path to the dataset files
            win_size (int): Size of the sliding window
            step (int): Step size for the sliding window (default: 100)
            mode (str): 'train' or 'test' mode
        """
        # Initialize the base class with window parameters
        super(SMDDataset, self).__init__(win_size, step, mode)

        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
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
