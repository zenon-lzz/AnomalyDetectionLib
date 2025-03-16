"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-16
@Description: SMD (Server Machine Dataset) Dataloader
    This module implements a PyTorch Dataset for the Server Machine Dataset,
    which contains server performance metrics for anomaly detection.
    
    Features:
    - Sliding window-based data sampling
    - Data standardization
    - Train/Val/Test split handling
    - Larger default step size (100) for efficient processing
==================================================
"""
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SMDLoader(Dataset):
    """
    PyTorch Dataset for Server Machine Dataset (SMD).
    
    Features:
    - Handles multivariate server performance metrics
    - Implements sliding window with larger step size
    - Performs data standardization
    - Supports train/validation/test splits
    
    Note:
    - Default step size is 100 (larger than other datasets)
    - Uses 80-20 split for train-validation
    """

    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        """
        Initialize SMD dataloader.
        
        Args:
            args: Configuration arguments
            root_path: Path to SMD dataset directory
            win_size: Size of sliding window
            step: Step size for window sliding (default: 100)
            flag: Dataset split identifier ("train"/"val"/"test")
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size

        # Initialize and fit StandardScaler on training data
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Load and transform test data
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data

        # Create validation split (last 20% of training data)
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        # Load anomaly labels for test set
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        """
        Calculate number of available windows based on step size.
        
        Returns:
            int: Number of windows in the dataset
        
        Note:
            Uses larger step size (100) for efficiency in processing
            server metrics data
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
        Get a window of server metrics and corresponding labels.
        
        Args:
            index: Index of the window
        
        Returns:
            tuple: (window_data, window_labels)
                - window_data: Normalized metrics [win_size, n_metrics]
                - window_labels: Anomaly labels [win_size]
        
        Note:
            - Train/val use initial test labels
            - Test uses aligned test labels
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
