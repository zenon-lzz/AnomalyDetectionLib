"""
=================================================
@Author: Zenon
@Date: 2025-03-13
@Description: HexagonML (UCR) dataset Time Series Dataset preprocess module
Details refer to the paper "The UCR Time Series Classification Archive"
==================================================
"""
import os
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure

from tsadlib import log
from .base import BaseDataset
from ..plotting import LinePlot


class UCRDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize UCR dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files
        """
        super().__init__(data_dir, save_dir)

        self.train_data: Dict[int, np.ndarray] = {}
        self.test_data: Dict[int, np.ndarray] = {}
        self.labels_data: Dict[int, np.ndarray] = {}
        self.dataset_ids: List[int] = []

    def _parse_filename(self, filename: str) -> Tuple[int, List[int]]:
        """Parse dataset number and values from filename.
        
        Args:
            filename: Name of the data file
            
        Returns:
            Tuple containing dataset number and list of values
        """
        vals = filename.split('.')[0].split('_')
        dataset_num = int(vals[0])
        values = [int(i) for i in vals[-3:]]
        return dataset_num, values

    def _load_and_normalize(self, file_path: str) -> np.ndarray:
        """Load and normalize data from file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            np.ndarray: Normalized data
        """
        data = np.genfromtxt(file_path, dtype=np.float64, delimiter=',')
        min_val, max_val = np.min(data), np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def load_data(self) -> None:
        """Load UCR dataset from txt files."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found at {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.txt'):
                continue

            file_path = os.path.join(self.data_dir, filename)
            dataset_num, values = self._parse_filename(filename)

            if dataset_num not in self.dataset_ids:
                self.dataset_ids.append(dataset_num)

            log.info(f"Processing dataset {dataset_num} from {filename}")

            # Load and normalize data
            data = self._load_and_normalize(file_path)

            # Split into train and test
            train = data[:values[0]]
            test = data[values[0]:]

            # Create labels
            labels = np.zeros_like(test)
            labels[values[1] - values[0]:values[2] - values[0]] = 1

            # Reshape data
            self.train_data[dataset_num] = train.reshape(-1, 1)
            self.test_data[dataset_num] = test.reshape(-1, 1)
            self.labels_data[dataset_num] = labels.reshape(-1, 1)

    def preprocess(self, is_normalize: bool = False) -> None:
        """Preprocess UCR dataset.
        
        Note: Data is already normalized during loading, so is_normalize is ignored.
        """
        log.info("Preprocessing UCR data...")

        if not self.train_data:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        self.train = self.train_data
        self.test = self.test_data
        self.labels = self.labels_data

        logger.info("UCR data preprocessing completed")

    def visualize(self) -> List[Figure]:
        """Visualize each dataset's time series."""
        figures = []

        for dataset_num in self.dataset_ids:
            logger.info(f'Plotting time series for dataset {dataset_num}')

            # Create visualization for training data
            figures.append(LinePlot.plot_time_series(
                self.train[dataset_num],
                column_names=['Value'],
                title=f'Dataset {dataset_num} - Training'
            ))

            # Create visualization for test data with anomaly labels
            figures.append(LinePlot.plot_time_series(
                self.test[dataset_num],
                column_names=['Value'],
                title=f'Dataset {dataset_num} - Test',
                labels_data=self.labels[dataset_num]
            ))

        return figures

    def get_statistics(self) -> List[Dict]:
        """Get dataset statistics for all datasets."""
        if not self.train:
            logger.error("Data not processed yet")
            return []

        stats = []
        for dataset_num in self.dataset_ids:
            stats.append({
                'dataset_id': dataset_num,
                'train_samples': len(self.train[dataset_num]),
                'test_samples': len(self.test[dataset_num]),
                'dimensions': 1,
                'anomaly_rate': float(np.mean(self.labels[dataset_num])),
                'anomaly_samples': int(np.sum(self.labels[dataset_num]))
            })
        return stats
