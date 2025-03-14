"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-13
@Description: MSL (Mars Science Laboratory) and SMAP (Soil Moisture Active Passive) Dataset preprocess module
Details refer to the paper "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
==================================================
"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

from tsadlib import logger
from tsadlib.utils.scaler import minmax_scaler
from .base import BaseDataset
from ..plotting import LinePlot


class MSLSMAPDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None,
                 spacecraft: str = 'MSL'):
        """Initialize MSL/SMAP dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files
            spacecraft: Which spacecraft data to process ('MSL' or 'SMAP')
        """
        super().__init__(data_dir, save_dir)

        self.spacecraft = spacecraft
        self.metadata: DataFrame | None = None
        self.train_data: Dict[str, np.ndarray] = {}
        self.test_data: Dict[str, np.ndarray] = {}
        self.labels_data: Dict[str, np.ndarray] = {}
        self.channel_ids: List[str] = []

    def load_data(self) -> None:
        """Load MSL/SMAP dataset from numpy files and metadata CSV."""
        # Load metadata file
        metadata_file = os.path.join(self.data_dir, 'labeled_anomalies.csv')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

        self.metadata = pd.read_csv(metadata_file)
        # Filter by spacecraft
        self.metadata = self.metadata[self.metadata['spacecraft'] == self.spacecraft]
        self.channel_ids = self.metadata['chan_id'].to_numpy().tolist()

        # Load data for each channel
        for channel_id in self.channel_ids:
            train_file = os.path.join(self.data_dir, 'train', f'{channel_id}.npy')
            test_file = os.path.join(self.data_dir, 'test', f'{channel_id}.npy')

            if not all(os.path.exists(f) for f in [train_file, test_file]):
                logger.warning(f"Data files not found for channel {channel_id}")
                continue

            logger.info(f"Loading data from channel {channel_id}")
            self.train_data[channel_id] = np.load(train_file)
            self.test_data[channel_id] = np.load(test_file)

            # Process anomaly labels
            labels = np.zeros_like(self.test_data[channel_id])
            anomaly_sequences = self.metadata[
                self.metadata['chan_id'] == channel_id
                ]['anomaly_sequences'].values[0]

            # Parse anomaly indices
            indices = anomaly_sequences.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]

            # Mark anomaly regions
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1

            self.labels_data[channel_id] = labels

    def preprocess(self, is_normalize: bool = True) -> None:
        """Preprocess MSL/SMAP dataset with normalization."""
        logger.info(f"Preprocessing {self.spacecraft} data...")

        if not self.train_data:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        self.train = {}
        self.test = {}
        self.labels = {}

        for channel_id in self.channel_ids:
            train_data = self.train_data[channel_id]
            test_data = self.test_data[channel_id]

            if is_normalize:
                # Min-max normalization
                train_data, min_vals, max_vals = minmax_scaler(train_data)
                test_data, _, _ = minmax_scaler(test_data, min_vals, max_vals)

            # Store processed data
            self.train[channel_id] = train_data
            self.test[channel_id] = test_data
            self.labels[channel_id] = self.labels_data[channel_id]

        logger.info(f"{self.spacecraft} data preprocessing completed")

    def save(self) -> None:
        """Save processed data for each channel separately."""

        if self.save_dir is None:
            logger.error("The save_dir is not specified, operation is skipped.")
            return

        os.makedirs(self.save_dir, exist_ok=True)

        for channel_id in self.channel_ids:
            np.save(os.path.join(self.save_dir, f'{channel_id}_train.npy'), self.train[channel_id])
            np.save(os.path.join(self.save_dir, f'{channel_id}_test.npy'), self.test[channel_id])
            np.save(os.path.join(self.save_dir, f'{channel_id}_labels.npy'), self.labels[channel_id])

        logger.info(f"Data has been saved to {self.save_dir}")

    def visualize(self) -> List[Figure]:
        """Visualize each channel's time series."""
        figures = []

        for channel_id in self.channel_ids:
            logger.info(f'Plotting time series for channel {channel_id}')

            train_data = self.train[channel_id]
            test_data = self.test[channel_id]
            labels_data = self.labels[channel_id]

            # Get feature names from metadata if available
            column_names = [f'Feature_{i}' for i in range(train_data.shape[1])]

            # Create visualization for training data
            figures.append(LinePlot.plot_time_series(
                train_data,
                column_names=column_names,
                title=f'{self.spacecraft} Channel {channel_id} - Training'
            ))

            # Create visualization for test data with anomaly labels
            figures.append(LinePlot.plot_time_series(
                test_data,
                column_names=column_names,
                title=f'{self.spacecraft} Channel {channel_id} - Test',
                labels_data=labels_data
            ))

        return figures

    def get_statistics(self) -> List[Dict]:
        """Get dataset statistics for all channels."""
        if not self.train:
            logger.error("Data not processed yet")
            return []

        stats = []

        for channel_id in self.channel_ids:
            stats.append({
                'channel_id': channel_id,
                'train_samples': len(self.train[channel_id]),
                'test_samples': len(self.test[channel_id]),
                'dimensions': self.train[channel_id].shape[1],
                'anomaly_rate': float(np.mean(self.labels[channel_id])),
                'anomaly_samples': int(np.sum(self.labels[channel_id]))
            })
        return stats
