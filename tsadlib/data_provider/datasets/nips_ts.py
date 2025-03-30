"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: NIPS Time Series Datasets Loader
    This module implements loaders for multiple NIPS benchmark time series datasets:
    - Water: Water treatment plant sensor data
    - Swan: Industrial process monitoring data  
    - Creditcard: Synthetic credit card transaction data
==================================================
"""
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class NIPSTSWaterDataset(BaseDataset):
    """Water treatment plant sensor data loader for anomaly detection.
    
    Inherits from BaseDataset and implements water-specific data loading.
    
    Args:
        root_path (str): Path to directory containing dataset files
        win_size (int): Sliding window size for time series segmentation
        step (int): Stride between windows (default: 1)
        mode (str): 'train' or 'test' mode (default: 'train')
        
    Attributes:
        scaler (StandardScaler): Scaler fitted on training data
        train (np.ndarray): Normalized training data
        test (np.ndarray): Normalized test data (mode='test')
        test_labels (np.ndarray): Anomaly labels (mode='test')
    """

    def __init__(self, root_path, win_size, step=1, mode='train'):
        """Initialize dataset and load/preprocess data.
        
        Processing Pipeline:
        1. Load numpy array from .npy file
        2. Fit scaler on training data
        3. Apply standardization
        """
        super().__init__(win_size, step, mode)

        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, 'NIPS_TS_Water_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':
            self.train = data
        elif mode == 'test':
            test_data = np.load(os.path.join(root_path, 'NIPS_TS_Water_test.npy'))
            self.test = self.scaler.transform(test_data)
            self.test_labels = np.load(os.path.join(root_path, 'NIPS_TS_Water_test_label.npy'))


class NIPSTSSwanDataset(BaseDataset):
    """Industrial process monitoring data loader.
    
    See NIPSTSWaterDataset for full parameter and attribute documentation.
    Dataset-specific implementation for Swan benchmark data.
    """

    def __init__(self, root_path, win_size, step=1, mode='train'):

        super().__init__(win_size, step, mode)

        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, 'NIPS_TS_Swan_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':

            self.train = data
        elif mode == 'test':

            test_data = np.load(os.path.join(root_path, 'NIPS_TS_Swan_test.npy'))
            self.test = self.scaler.transform(test_data)

            self.test_labels = np.load(os.path.join(root_path, 'NIPS_TS_Swan_test_label.npy'))


class NIPSTSCCardDataset(BaseDataset):
    """Synthetic credit card transaction data loader.
    
    See NIPSTSWaterDataset for full parameter and attribute documentation.  
    Dataset-specific implementation for Creditcard benchmark data.
    """

    def __init__(self, root_path, win_size, step=1, mode='train'):

        super().__init__(win_size, step, mode)

        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, 'NIPS_TS_Creditcard_train.npy'))
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        if mode == 'train':

            self.train = data
        elif mode == 'test':

            test_data = np.load(os.path.join(root_path, 'NIPS_TS_Creditcard_test.npy'))
            self.test = self.scaler.transform(test_data)

            self.test_labels = np.load(os.path.join(root_path, 'NIPS_TS_Creditcard_test_label.npy'))
