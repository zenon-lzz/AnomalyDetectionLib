"""
=================================================
@Author: Zenon
@Date: 2025-03-13
@Description: SWaT (Secure Water Treatment) Dataset preprocess module
Details refer to the paper "SWaT: a water treatment testbed for research and training on ICS security"
==================================================
"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tsadlib import log
from .base import BaseDataset
from ..plotting import LinePlot
from ..utils.scaler import minmax_scaler_global


class SWaTDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize SWaT dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files
        """
        super().__init__(data_dir, save_dir)

        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.labels_data: pd.DataFrame = None

    def load_data(self) -> None:
        """Load SWaT dataset from JSON file."""
        json_file = os.path.join(self.data_dir, 'series.json')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Data file not found at {json_file}")

        log.info("Loading SWaT data from JSON file...")
        df = pd.read_json(json_file, lines=True)

        # Extract training and test data
        self.train_data = df[['val']][3000:6000]
        self.test_data = df[['val']][7000:12000]
        self.labels_data = df[['noti']][7000:12000]

    def preprocess(self, is_normalize: bool = True) -> None:
        """Preprocess SWaT dataset with normalization."""
        log.info("Preprocessing SWaT data...")

        if self.train_data is None or self.test_data is None:
            log.error("Data not loaded yet, please call load_data() first")
            return

        # Convert to numpy arrays
        train = self.train_data.values
        test = self.test_data.values

        if is_normalize:
            # Global min-max normalization
            train, min_val, max_val = minmax_scaler_global(train)
            test, _, _ = minmax_scaler_global(test, min_val, max_val)

        # Convert labels to binary values
        labels = self.labels_data.values + 0

        self.train = train
        self.test = test
        self.labels = labels

        log.info("SWaT data preprocessing completed")

    def visualize(self) -> List[Figure]:
        """Visualize the time series data."""
        figures = []

        if self.train is None or self.test is None:
            log.error("Data not processed yet")
            return figures

        # Create visualization for training data
        figures.append(LinePlot.plot_time_series(
            self.train,
            column_names=['Value'],
            title='SWaT Training Data'
        ))

        # Create visualization for test data with anomaly labels
        figures.append(LinePlot.plot_time_series(
            self.test,
            column_names=['Value'],
            title='SWaT Test Data',
            labels_data=self.labels
        ))

        return figures

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.train is None or self.test is None:
            log.error("Data not processed yet")
            return {}

        stats = {
            'train_samples': len(self.train),
            'test_samples': len(self.test),
            'dimensions': self.train.shape[1],
            'anomaly_rate': float(np.mean(self.labels)),
            'anomaly_samples': int(np.sum(self.labels))
        }

        return stats
