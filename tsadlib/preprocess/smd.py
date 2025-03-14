"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-13
@Description: SMD (Server Machine Dataset) Dataset preprocess module
Details refer to the paper "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network"
==================================================
"""
import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure

from tsadlib import logger
from tsadlib.utils.scaler import minmax_scaler
from .base import BaseDataset
from ..plotting import LinePlot


class SMDDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize SMD dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files
        """
        super().__init__(data_dir, save_dir)

        self.train_data: Dict[str, np.ndarray] = {}
        self.test_data: Dict[str, np.ndarray] = {}
        self.labels_data: Dict[str, np.ndarray] = {}
        self.machine_ids: List[str] = []

    def _load_data_file(self, category: str, filename: str) -> np.ndarray:
        """Load data from txt file.
        
        Args:
            category: Data category (train/test)
            filename: Name of the file to load
            
        Returns:
            np.ndarray: Loaded data
        """
        file_path = os.path.join(self.data_dir, category, filename)
        data = np.genfromtxt(file_path, dtype=np.float64, delimiter=',')
        logger.info(f"Loaded {category} data from {filename}: shape {data.shape}")
        return data

    def _load_labels(self, filename: str, data_shape: Tuple[int, int]) -> np.ndarray:
        """Load and process anomaly labels.
        
        Args:
            filename: Name of the label file
            data_shape: Shape of the corresponding data array
            
        Returns:
            np.ndarray: Processed labels
        """
        labels = np.zeros(data_shape)
        label_file = os.path.join(self.data_dir, 'interpretation_label', filename)

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            pos, values = line.split(':')[0], line.split(':')[1].split(',')
            start, end = map(int, pos.split('-'))
            indices = [int(i) - 1 for i in values]
            labels[start - 1:end - 1, indices] = 1

        logger.info(f"Processed labels for {filename}: shape {labels.shape}")
        return labels

    def load_data(self) -> None:
        """Load SMD dataset from txt files."""
        train_dir = os.path.join(self.data_dir, "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training data directory not found at {train_dir}")

        for filename in os.listdir(train_dir):
            if not filename.endswith('.txt'):
                continue

            machine_id = filename.replace('.txt', '')
            self.machine_ids.append(machine_id)

            # Load train and test data
            self.train_data[machine_id] = self._load_data_file('train', filename)
            self.test_data[machine_id] = self._load_data_file('test', filename)

            # Load and process labels
            self.labels_data[machine_id] = self._load_labels(
                filename,
                self.test_data[machine_id].shape
            )

    def preprocess(self, is_normalize: bool = True) -> None:
        """Preprocess SMD dataset with normalization."""
        logger.info("Preprocessing SMD data...")

        if not self.train_data:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        self.train = {}
        self.test = {}
        self.labels = {}

        for machine_id in self.machine_ids:
            train_data = self.train_data[machine_id]
            test_data = self.test_data[machine_id]

            if is_normalize:
                # Min-max normalization
                train_data, min_vals, max_vals = minmax_scaler(train_data)
                test_data, _, _ = minmax_scaler(test_data, min_vals, max_vals)

            # Store processed data
            self.train[machine_id] = train_data
            self.test[machine_id] = test_data
            self.labels[machine_id] = self.labels_data[machine_id]

        logger.info("SMD data preprocessing completed")

    def save(self) -> None:
        """Save processed data for each machine separately."""
        if self.save_dir is None:
            logger.error("The save_dir is not specified, operation is skipped.")
            return

        os.makedirs(self.save_dir, exist_ok=True)

        for machine_id in self.machine_ids:
            np.save(os.path.join(self.save_dir, f'{machine_id}_train.npy'), self.train[machine_id])
            np.save(os.path.join(self.save_dir, f'{machine_id}_test.npy'), self.test[machine_id])
            np.save(os.path.join(self.save_dir, f'{machine_id}_labels.npy'), self.labels[machine_id])

        logger.info(f"Data has been saved to {self.save_dir}")

    def visualize(self) -> List[Figure]:
        """Visualize each machine's time series."""
        figures = []

        for machine_id in self.machine_ids:
            logger.info(f'Plotting time series for machine {machine_id}')

            train_data = self.train[machine_id]
            test_data = self.test[machine_id]
            labels_data = self.labels[machine_id]

            # Create feature names
            column_names = [f'Feature_{i}' for i in range(train_data.shape[1])]

            # Create visualization for training data
            figures.append(LinePlot.plot_time_series(
                train_data,
                column_names=column_names,
                title=f'Machine {machine_id} - Training'
            ))

            # Create visualization for test data with anomaly labels
            figures.append(LinePlot.plot_time_series(
                test_data,
                column_names=column_names,
                title=f'Machine {machine_id} - Test',
                labels_data=labels_data
            ))

        return figures

    def get_statistics(self) -> List[Dict]:
        """Get dataset statistics for all machines."""
        if not self.train:
            logger.error("Data not processed yet")
            return []

        stats = []
        for machine_id in self.machine_ids:
            stats.append({
                'machine_id': machine_id,
                'train_samples': len(self.train[machine_id]),
                'test_samples': len(self.test[machine_id]),
                'dimensions': self.train[machine_id].shape[1],
                'anomaly_rate': float(np.mean(self.labels[machine_id])),
                'anomaly_samples': int(np.sum(self.labels[machine_id]))
            })
        return stats
