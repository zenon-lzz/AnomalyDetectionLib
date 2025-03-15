"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-12
@Description：MBA (MIT-BIH Supraventricular Arrhythmia Database) dataset preprocess module
Details refer to the paper "The impact of the MIT-BIH arrhythmia database"
==================================================
"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

from tsadlib import logger
from .base import BaseDataset
from ..plotting import LinePlot
from ..utils.scaler import minmax_scaler_column_wise


class MBADataset(BaseDataset):
    """MBA dataset class for loading and preprocessing MIT-BIH Supraventricular Arrhythmia data.
    
    This class handles loading the raw ECG data from Excel files, preprocessing including normalization,
    visualization of the signals, and computing dataset statistics.
    """

    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize MBA dataset.
        
        Args:
            data_dir: Directory containing the raw Excel data files
            save_dir: Directory to save processed data files. If None, saving is disabled.
        """
        super().__init__(data_dir, save_dir)

        self.train_data: DataFrame = None  # Raw training data
        self.test_data: DataFrame = None  # Raw test data
        self.labels_data: DataFrame = None  # Raw anomaly labels

    def load_data(self) -> None:
        """Load MBA dataset from Excel files.
        
        Loads train.xlsx, test.xlsx and labels.xlsx files from data_dir.
        Raises FileNotFoundError if any file is missing.
        """
        train_file = os.path.join(self.data_dir, 'train.xlsx')
        test_file = os.path.join(self.data_dir, 'test.xlsx')
        labels_file = os.path.join(self.data_dir, 'labels.xlsx')

        if not all(os.path.exists(f) for f in [train_file, test_file, labels_file]):
            raise FileNotFoundError(f"Complete MBA data files not found in {self.data_dir}")

        # Load data from Excel files
        self.train_data = pd.read_excel(train_file)
        self.test_data = pd.read_excel(test_file)
        self.labels_data = pd.read_excel(labels_file)

    def preprocess(self,
                   is_normalize: bool = True) -> None:
        """Preprocess the MBA dataset.
        
        Converts data to float arrays and optionally normalizes using min-max scaling.
        Creates binary labels array with window of ±20 samples around labeled anomalies.
        
        Args:
            is_normalize: Whether to apply min-max normalization. Defaults to True.
        """
        logger.info("Preprocessing MBA data...")

        if self.train_data is None:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        # Extract numerical data, skipping first row/column
        train = self.train_data.values[1:, 1:].astype(float)
        test = self.test_data.values[1:, 1:].astype(float)

        # Normalize data using min-max scaling
        if is_normalize:
            logger.info("Normalizing data using min-max strategy")
            train, min_a, max_a = minmax_scaler_column_wise(train)
            test, _, _ = minmax_scaler_column_wise(test, min_a, max_a)

        # Create binary labels array with ±20 sample window
        ls = self.labels_data.values[:, 1].astype(int)
        labels = np.zeros_like(test)

        for i in range(-20, 20):
            labels[ls + i, :] = 1

        self.train = train
        self.test = test
        self.labels = labels
        logger.info("MBA data preprocessing completed")

    def visualize(self) -> List[Figure]:
        """Visualize the MBA dataset.
        
        Creates line plots for both training and test data.
        Test data plot includes anomaly labels highlighted in red.
        
        Returns:
            List of matplotlib Figures containing the plots
        """
        columns = [column + ' (mV)' for column in self.train_data.columns[1:]]
        figures = []
        # x_data = self.train_data.to_numpy()[1:, 0][100:1000].astype(int)
        x_data = None
        # Plot training set
        logger.info('Plotting time series for each dimension of MBA training set')
        train_data = self.train_data.to_numpy()[1:, 1:]
        figures.append(LinePlot.plot_time_series(train_data, x_data=x_data, column_names=columns, title='Training set'))

        # Plot test set with anomaly labels
        logger.info('Plotting time series for each dimension of MBA test set with anomaly labels')
        test_data = self.test_data.to_numpy()[1:, 1:]
        labels_data = np.zeros(len(test_data))
        labels_data[self.labels_data.to_numpy()[:, 1].astype(int)] = 1
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
            anomaly_ratio = np.mean(self.labels)
            stats['anomaly_rate'] = float(anomaly_ratio)
            stats['anomaly_samples'] = np.sum(np.any(self.labels, axis=1))

        return stats
