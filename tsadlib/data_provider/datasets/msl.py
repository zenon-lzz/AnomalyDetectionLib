'''
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: MSL (Mars Science Laboratory) Dataset
    This module provides a PyTorch Dataset implementation for the MSL anomaly detection dataset.
    Features:
    - Sliding window-based data loading
    - Data standardization
    - Train/Val/Test split handling
    - Configurable step size for window sliding
==================================================
'''
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from tsadlib import logger


class MSLDataset(Dataset):
    """
    PyTorch Dataset for MSL anomaly detection data.
    
    Features:
    - Loads and preprocesses MSL telemetry data
    - Implements sliding window mechanism
    - Supports train/validation/test splits
    - Standardizes data using StandardScaler
    
    Data Structure:
    - Training data: Normalized telemetry data
    - Test data: Separate test set with labels
    - Validation: Last 20% of training data
    """

    def __init__(self, args, root_path, win_size, step=1, flag='train'):
        """
        Initialize the MSL dataset.
        
        Args:
            args: Configuration arguments
            root_path: Path to MSL dataset files
            win_size: Size of sliding window
            step: Step size for sliding window (default: 1)
            flag: Dataset split to use ('train'/'val'/'test')
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size

        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, 'MSL_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Load and transform test data
        test_data = np.load(os.path.join(root_path, 'MSL_test.npy'))
        self.test = self.scaler.transform(test_data)
        self.train = data

        # Create validation split (last 20% of training data)
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        # Load test labels
        self.test_labels = np.load(os.path.join(root_path, 'MSL_test_label.npy'))
        logger.info(f'test set\'s shape: {self.test.shape}')
        logger.info(f'train set\'s shape: {self.train.shape}')

    def __len__(self):
        """
        Calculate number of samples based on window size and step.
        
        Returns:
            int: Number of available windows in the dataset
        """
        if self.flag == 'train':
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            # Special case for non-overlapping windows
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a single sample using sliding window.
        
        Args:
            index: Index of the window
        
        Returns:
            tuple: (window_data, window_labels)
                - window_data: Normalized data window [win_size, features]
                - window_labels: Corresponding labels [win_size]
        
        Note:
            - For train/val: Uses same initial labels
            - For test: Uses corresponding test labels
            - Special case uses non-overlapping windows
        """
        index = index * self.step
        if self.flag == 'train':
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            # Special case: non-overlapping windows
            start_idx = index // self.step * self.win_size
            end_idx = start_idx + self.win_size
            return np.float32(self.test[start_idx:end_idx]), np.float32(
                self.test_labels[start_idx:end_idx])
