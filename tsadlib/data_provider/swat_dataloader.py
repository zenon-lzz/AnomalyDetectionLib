"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-16
@Description: SWaT (Secure Water Treatment) Dataset Loader
    This module implements a PyTorch Dataset for the SWaT dataset,
    which contains industrial control system data from a water treatment plant.
    
    Features:
    - CSV-based data loading
    - Sliding window approach for time series segmentation
    - Data standardization
    - Train/Val/Test split handling
==================================================
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SWATSegLoader(Dataset):
    """
    PyTorch Dataset for SWaT (Secure Water Treatment) data.
    
    Features:
    - Loads CSV data from industrial control system
    - Implements sliding window for time series segmentation
    - Performs data standardization
    - Supports train/validation/test splits
    
    Dataset Structure:
    - Training data: Normal operation data
    - Test data: Contains both normal and attack scenarios
    - Labels: Binary indicators for normal/attack states
    """

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        """
        Initialize SWaT dataloader.
        
        Args:
            args: Configuration arguments
            root_path: Path to SWaT dataset directory
            win_size: Size of sliding window
            step: Step size for window sliding (default: 1)
            flag: Dataset split identifier ("train"/"val"/"test")
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load CSV data files
        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))

        # Extract labels from the last column of test data
        labels = test_data.values[:, -1:]

        # Remove label column from feature data
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        # Standardize data using training set statistics
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        # Store processed data
        self.train = train_data
        self.test = test_data

        # Create validation split (last 20% of training data)
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        # Store test labels
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Calculate number of windows in the dataset.
        
        Returns:
            int: Number of available windows based on window size and step
        
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
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        """
        Get a window of sensor data and corresponding labels.
        
        Args:
            index: Index of the window
        
        Returns:
            tuple: (window_data, window_labels)
                - window_data: Normalized sensor readings [win_size, n_sensors]
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
