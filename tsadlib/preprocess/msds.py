"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-12
@Description：Multi-Source Distributed System (MSDS) Dataset preprocess module
Details refer to the paper "Multi-source distributed system data for AI-powered analytics"

TODO: No Anomalies in Test set.
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


class MSDSDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        super().__init__(data_dir, save_dir)

        self.train_data: DataFrame = None  # Raw training data
        self.test_data: DataFrame = None  # Raw test data
        self.labels_data: DataFrame = None  # Raw anomaly labels

    def load_data(self) -> None:
        train_file = os.path.join(self.data_dir, 'train.csv')
        test_file = os.path.join(self.data_dir, 'test.csv')
        labels_file = os.path.join(self.data_dir, 'labels.csv')

        if not all(os.path.exists(f) for f in [train_file, test_file, labels_file]):
            raise FileNotFoundError(f"Complete MSDS data files not found in {self.data_dir}")

        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.labels_data = pd.read_csv(labels_file)

    def preprocess(self,
                   is_normalize: bool = True) -> None:
        logger.info("Preprocessing MSDS data...")

        if self.train_data is None:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        # Extract numerical data, skipping first row/column
        train = self.train_data.to_numpy()[::5, 1:]
        test = self.test_data.to_numpy()[::5, 1:]

        # Normalize data using min-max scaling
        if is_normalize:
            logger.info("Normalizing data using min-max strategy")
            _, min_a, max_a = minmax_scaler(np.concatenate((train, test), axis=0))
            train, _, _ = minmax_scaler(train, min_a, max_a)
            test, _, _ = minmax_scaler(test, min_a, max_a)

        labels = self.labels_data.to_numpy()[::1, 1:]

        self.train = train
        self.test = test
        self.labels = labels
        logger.info("MSDS data preprocessing completed")

    def visualize(self) -> List[Figure]:
        columns = self.train_data.columns[1:]
        figures = []
        # x_data = self.train_data.to_numpy()[::5, 1:][100:1000]
        x_data = None
        # Plot training set
        logger.info('Plotting time series for each dimension of MSDS training set')
        train_data = self.train_data.to_numpy()[::5, 1:]
        figures.append(LinePlot.plot_time_series(train_data, x_data=x_data, column_names=columns, title='Training set'))

        # Plot test set with anomaly labels
        logger.info('Plotting time series for each dimension of MSDS test set with anomaly labels')
        test_data = self.test_data.to_numpy()[::5, 1:]
        labels_data = self.labels_data.to_numpy()[::1, 1:]
        figures.append(LinePlot.plot_time_series(test_data, x_data=x_data, column_names=columns,
                                                 title='Test set',
                                                 labels_data=labels_data
                                                 ))
        return figures

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Computes basic statistics about the dataset dimensions and anomaly distribution.

        Returns:
            Dict: Dictionary containing dataset statistics including:
                - Train and test set dimensions
                - Anomaly rate in test set
                - Total number of anomaly samples
        """
        if self.train is None or self.test is None:
            logger.error("Data not loaded yet")
            return {}

        stats = {
            'train_samples': self.train.shape[0],
            'dimensions': self.train.shape[1],
            'test_samples': self.test.shape[0]
        }

        if self.labels is not None:
            # Calculate anomaly ratio in test set
            # Calculate anomaly ratio based on any anomaly across dimensions at each timestamp
            anomaly_ratio = np.mean(np.any(self.labels == 1, axis=1))
            stats['anomaly_rate'] = float(anomaly_ratio)
            stats['anomaly_samples'] = np.sum(np.any(self.labels, axis=1))

        return stats
