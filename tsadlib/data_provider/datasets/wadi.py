"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: WADI (Water Distribution) Dataset Loader
    This module implements the WADI dataset loader for industrial control system anomaly detection.
    The dataset contains sensor readings from a water distribution system with labeled attack periods.
==================================================
"""
import os

import pandas as pd

from .base import BaseDataset
from ... import PreprocessScalerEnum, ConfigType


class WADIDataset(BaseDataset):
    """Water Distribution Anomaly Detection dataset loader.
    
    Inherits from BaseDataset and implements WADI-specific data loading and preprocessing.

    """

    def __init__(self, root_path, args: ConfigType, mode, scaler: PreprocessScalerEnum):
        """Initialize dataset and load/preprocess data.
        
        Data Processing Pipeline:
        1. Load CSV files and clean column names
        2. Remove metadata columns and all-NaN columns
        3. Handle missing values via interpolation
        4. Standardize data using training statistics
        """
        super().__init__(args.window_size, args.window_stride, mode)

        # Load and clean training data
        train_data = pd.read_csv(os.path.join(root_path, 'WADI_14days.csv'))
        train_data.columns = [col.strip(' ') for col in train_data.columns]  # Remove whitespace in column names

        # Identify and remove non-feature columns and all-NaN columns
        train_nan_columns = {col for col in train_data.columns if train_data[col].isna().all()}
        train_data = train_data.drop(['Row', 'Date', 'Time'] + list(train_nan_columns), axis=1)

        # Handle missing values via linear interpolation and backfill
        train_data = train_data.interpolate().bfill().to_numpy()

        # Standardize data using training statistics
        self.set_scaler(scaler)
        self.scaler.fit(train_data)
        data = self.scaler.transform(train_data)

        if mode == 'train':
            self.train = data
        elif mode == 'test':
            # Load and preprocess test data
            test_data = pd.read_csv(os.path.join(root_path, 'WADI_attackdataLABLE.csv'), header=1)
            test_data.columns = [col.strip(' ') for col in test_data.columns]
            
            # Process attack labels (convert to binary: 1=attack, 0=normal)
            self.test_labels = test_data["Attack LABLE (1:No Attack, -1:Attack)"].apply(
                lambda x: 0 if x == 1 else 1
            ).to_numpy()
            test_data.drop(["Attack LABLE (1:No Attack, -1:Attack)"], axis=1)

            # Apply same preprocessing as training data
            test_nan_columns = {col for col in test_data.columns if test_data[col].isna().all()}
            test_data = test_data.drop(['Row', 'Date', 'Time'] + list(test_nan_columns), axis=1)
            test_data = test_data.interpolate().bfill().to_numpy()
            self.test = self.scaler.transform(test_data)
