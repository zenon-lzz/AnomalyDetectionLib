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

from .base import BaseDataset
from ... import ConfigType, PreprocessScalerEnum


class PSMDataset(BaseDataset):
    """Pooled Server Metrics dataset loader for anomaly detection.
    
    Inherits from BaseDataset and implements PSM-specific data loading and preprocessing.
    

    """

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        """Initialize dataset and load/preprocess data.
        
        Data Processing Pipeline:
        1. Load CSV files
        2. Fit scaler on training data
        3. Normalize all data
        4. Store based on mode
        """
        super().__init__(args.window_size, args.window_stride, mode)

        # Load and preprocess training data
        train_data = pd.read_csv(os.path.join(root_path, 'train.csv')).values[:, 1:]
        train_data = np.nan_to_num(train_data)
        self.set_scaler(scaler)
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
