"""
=================================================
@Author: Zenon
@Date: 2025-03-13
@Description: NAB (Numenta Anomaly Benchmark) Dataset preprocess module
Details refer to the paper "Evaluating Real-time Anomaly Detection Algorithms"
TODO: Training set and Test set is same.
==================================================
"""
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

from tsadlib import logger
from .base import BaseDataset
from ..plotting import LinePlot


class NABDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        super().__init__(data_dir, save_dir)

        self.label_dict = None
        self.train_data: Dict[str, DataFrame] = {}  # Raw training data for each file
        self.test_data: Dict[str, DataFrame] = {}  # Raw test data for each file
        self.labels_data: Dict[str, np.ndarray] = {}  # Raw anomaly labels for each file
        self.file_names: List[str] = []  # List of processed file names

    def load_data(self) -> None:
        """Load NAB dataset from CSV files and labels from JSON."""
        # Load labels from JSON file
        labels_file = os.path.join(self.data_dir, 'labels.json')
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found at {labels_file}")

        with open(labels_file) as f:
            self.label_dict = json.load(f)

        # Process each CSV file
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.csv'):
                continue

            file_path = os.path.join(self.data_dir, filename)
            logger.info(f"Loading data from {filename}")

            df = pd.read_csv(file_path)
            filename = filename.replace('.csv', '')
            self.file_names.append(filename)

            # Store raw data
            self.train_data[filename] = df
            self.test_data[filename] = df

            # Process labels
            labels = np.zeros(len(df), dtype=np.float64)
            label_key = f'realKnownCause/{filename}.csv'
            if label_key in self.label_dict:
                for timestamp in self.label_dict[label_key]:
                    tstamp = timestamp.replace('.000000', '')
                    index = np.where(((df['timestamp'] == tstamp).to_numpy() + 0) == 1)[0][0]
                    labels[index - 4:index + 4] = 1

            self.labels_data[filename] = labels

    def preprocess(self, is_normalize: bool = True) -> None:
        """Preprocess NAB dataset with normalization."""
        logger.info("Preprocessing NAB data...")

        if not self.train_data:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        self.train = {}
        self.test = {}
        self.labels = {}

        for filename in self.file_names:
            raw_data = self.train_data[filename].values[:, 1]

            if is_normalize:
                # Min-max normalization
                min_val, max_val = np.min(raw_data), np.max(raw_data)
                normalized_data = (raw_data - min_val) / (max_val - min_val)
            else:
                normalized_data = raw_data

            # Reshape data
            train_data = normalized_data.astype(float).reshape(-1, 1)
            test_data = normalized_data.astype(float).reshape(-1, 1)
            labels = self.labels_data[filename].reshape(-1, 1)

            # Store processed data
            self.train[filename] = train_data
            self.test[filename] = test_data
            self.labels[filename] = labels

        logger.info("NAB data preprocessing completed")

    def save(self) -> None:
        """Save processed data for each file separately."""
        if self.save_dir is None:
            logger.error("The save_dir is not specified, operation is skipped.")
            return

        os.makedirs(self.save_dir, exist_ok=True)

        for filename in self.file_names:
            np.save(os.path.join(self.save_dir, f'{filename}_train.npy'), self.train[filename])
            np.save(os.path.join(self.save_dir, f'{filename}_test.npy'), self.test[filename])
            np.save(os.path.join(self.save_dir, f'{filename}_labels.npy'), self.labels[filename])

        logger.info(f"Data has been saved to {self.save_dir}")

    def visualize(self) -> List[Figure]:
        """Visualize each time series in the NAB dataset."""
        figures = []

        for filename in self.file_names:
            # Plot training set
            logger.info(f'Plotting time series for {filename}')

            train_data = self.train_data[filename]
            test_data = self.test_data[filename]
            labels_data = self.labels[filename]

            column_names = train_data.columns[1]

            figures.append(LinePlot.plot_time_series(
                test_data.to_numpy()[:, 1:],
                column_names=column_names,
                title=f'Training and Test set - {filename}',
                labels_data=labels_data
            ))

        return figures

    def get_statistics(self) -> Dict | List[Dict]:
        """Get dataset statistics for all files."""
        if not self.train:
            logger.error("Data not processed yet")
            return {}

        stats = []
        for filename in self.file_names:
            stats.append({
                'file_name': filename,
                'dimensions': self.train[filename].shape[1],
                'samples': len(self.train[filename]),
                'anomaly_rate': float(np.mean(self.labels[filename])),
                'anomaly_samples': int(np.sum(self.labels[filename]))
            })

        return stats
