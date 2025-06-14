"""
=================================================
@Author: Zenon
@Date: 2025-03-27
@Description: Time Series Anomaly Detection Dataset Base Class
    This module provides a base dataset class for time series anomaly detection tasks.
    Specific dataset implementations like MSL and SMAP can inherit from this base class
    to reuse common data processing logic.
==================================================
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from tsadlib.configs.type import PreprocessScalerEnum


class BaseDataset(Dataset):

    def __init__(self, win_size, stride=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.stride = stride
        self.win_size = win_size

        # These attributes will be set by child classes
        self.train = None
        self.test = None
        self.test_labels = None
        self.scaler = None

    def set_scaler(self, scaler: PreprocessScalerEnum.STANDARD):
        if scaler is PreprocessScalerEnum.MINMAX:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def __len__(self):
        """
        Get the number of windows in the dataset.
        
        Returns:
            int: Number of available windows based on dataset mode
        """
        if self.mode == 'train':
            # Calculate number of windows for training data with sliding window
            return (self.train.shape[0] - self.win_size) // self.stride + 1
        elif self.mode == 'test':
            # Calculate number of windows for test data with sliding window
            return (self.test.shape[0] - self.win_size) // self.stride + 1
        else:
            # Special case for non-overlapping windows
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a window of data and corresponding labels.
        
        Args:
            index (int): Window index
            
        Returns:
            tuple: (data_window, label_window) as numpy arrays
        """
        index = index * self.stride
        if self.mode == 'train':
            # Use a zero array for training labels
            return np.float32(self.train[index:index + self.win_size]), np.zeros((self.win_size, 1), dtype=np.float32)
        elif self.mode == 'test':
            # Return window from test data with corresponding labels
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            # Special case: non-overlapping windows
            start_idx = index // self.stride * self.win_size
            end_idx = start_idx + self.win_size
            return np.float32(self.test[start_idx:end_idx]), np.float32(
                self.test_labels[start_idx:end_idx])
