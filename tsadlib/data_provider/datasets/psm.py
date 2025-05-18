"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: PSM (Pooled Server Metrics) Dataset Loader
    This module implements the PSM dataset loader for time series anomaly detection.
    The dataset contains metrics from pooled servers with labeled anomalies.
==================================================
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class PSMDataset(BaseDataset):
    """Pooled Server Metrics dataset loader for anomaly detection.
    
    Inherits from BaseDataset and implements PSM-specific data loading and preprocessing.
    
    Args:
        root_path (str): Path to directory containing PSM dataset files
        win_size (int): Sliding window size for time series segmentation
        step (int): Stride between windows (default: 1)
        mode (str): 'train' or 'test' mode (default: 'train')
    
    Attributes:
        scaler (StandardScaler): Scaler fitted on training data
        train (np.ndarray): Normalized training data (mode='train')
        test (np.ndarray): Normalized test data (mode='test')
        test_labels (np.ndarray): Anomaly labels (mode='test')
    """

    def __init__(self, root_path, win_size, step=1, mode="train"):
        """Initialize dataset and load/preprocess data.
        
        Data Processing Pipeline:
        1. Load CSV files
        2. Fit scaler on training data
        3. Normalize all data
        4. Store based on mode
        """
        super().__init__(win_size, step, mode)

        # Load and preprocess training data
        train_data = pd.read_csv(os.path.join(root_path, 'train.csv')).values[:, 1:]
        train_data = np.nan_to_num(train_data)
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)  # Fit scaler only on training data
        data = self.scaler.transform(train_data)

        if mode == 'train':
            self.train = data  # Store normalized training data
        elif mode == 'test':
            # Load and normalize test data
            test_data = pd.read_csv(os.path.join(root_path, 'test.csv')).values[:, 1:]
            test_data = np.nan_to_num(test_data)
            self.test = self.scaler.transform(test_data)

            # Load test labels (Note: Fixed typo from .cvs to .csv)
            self.test_labels = pd.read_csv(
                os.path.join(root_path, 'test_label.csv')
            ).values[:, 1:]
