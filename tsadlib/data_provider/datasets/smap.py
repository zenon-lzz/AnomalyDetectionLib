"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: SMAP (Soil Moisture Active Passive) Dataset Loader
    This module provides a PyTorch Dataset implementation for the SMAP anomaly detection dataset.
    Key Features:
    - Sliding window-based data sampling
    - Data standardization using StandardScaler
    - Train/Validation/Test split handling
    - Configurable window size and step size
==================================================
"""
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tsadlib.utils.logger import logger


class SMAPDataset(Dataset):
    """
    PyTorch Dataset implementation for SMAP telemetry data.
    
    Features:
    - Loads and preprocesses SMAP satellite telemetry data
    - Implements sliding window mechanism for sequence data
    - Handles train/validation/test splits
    - Performs data standardization
    
    Dataset Structure:
    - Training data: Main telemetry data for training
    - Validation: Last 20% of training data
    - Test data: Separate test set with anomaly labels
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize SMAP dataset.
        
        Args:
            args: Configuration arguments
            root_path: Path to SMAP dataset directory
            win_size: Size of sliding window
            step: Step size for window sliding (default: 1)
            flag: Dataset split identifier ("train"/"val"/"test")
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size

        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Load and transform test data
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data

        # Split validation set from training data (last 20%)
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]

        # Load test labels for anomaly detection
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        logger.info(f'test set\'s shape: {self.test.shape}')
        logger.info(f'train set\'s shape: {self.train.shape}')

    def __len__(self):
        """
        Calculate total number of samples based on window size and step.
        
        Returns:
            int: Number of available windows in the dataset
        
        Note:
            Different calculation for normal mode (step-based) and 
            special mode (non-overlapping windows)
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            # Special case: non-overlapping windows
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single sample window and its corresponding labels.
        
        Args:
            index: Index of the window to retrieve
        
        Returns:
            tuple: (window_data, window_labels)
                - window_data: Normalized data window [win_size, features]
                - window_labels: Corresponding anomaly labels [win_size]
        
        Note:
            - Train/val modes use initial test labels
            - Test mode uses aligned test labels
            - Special mode uses non-overlapping windows
        """
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
