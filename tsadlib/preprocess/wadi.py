"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-13
@Description: WADI (Water Distribution) Dataset preprocess module
Details refer to the paper "WADI: a water distribution testbed for research in the design of secure cyber physical systems"
TODO: No Anomalies
==================================================
"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tsadlib import logger
from .base import BaseDataset
from ..plotting import LinePlot


class WADIDataset(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize WADI dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files
        """
        super().__init__(data_dir, save_dir)

        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.labels_data: pd.DataFrame = None
        self.attack_labels: pd.DataFrame = None

    def _convert_to_numpy(self, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to numpy array, keeping only numerical columns."""
        # Skip the first 3 columns (Date, Time, and any other non-numerical columns)
        x = df.iloc[:, 3:].values[::10, :].astype(float)
        # - ptp is short for "peak to peak"
        # - Calculate the range of the array on axis 0 (maximum minus minimum)
        # - Is equivalent to x.mx (0) - x.mx (0)
        return (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-4)

    def load_data(self) -> None:
        """Load WADI dataset from CSV files."""
        # Load training data
        train_file = os.path.join(self.data_dir, 'WADI_14days.csv')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data file not found at {train_file}")

        logger.info("Loading training data...")
        self.train_data = pd.read_csv(train_file, skiprows=1000, nrows=int(2e5))

        # Load test data and attack labels
        test_file = os.path.join(self.data_dir, 'WADI_attackdata.csv')
        labels_file = os.path.join(self.data_dir, 'WADI_attacklabels.csv')

        if not all(os.path.exists(f) for f in [test_file, labels_file]):
            raise FileNotFoundError(f"Test or labels file not found in {self.data_dir}")

        logger.info("Loading test data and attack labels...")
        self.test_data = pd.read_csv(test_file)
        self.attack_labels = pd.read_csv(labels_file)

        # Clean data
        self._clean_data()

        # Process labels
        self._process_labels()

    def _clean_data(self) -> None:
        """Clean and preprocess the raw data."""
        # Handle missing values
        for df in [self.train_data, self.test_data]:
            df.dropna(how='all', inplace=True)
            df.fillna(0, inplace=True)

        # Process datetime in test data
        self.test_data['Time'] = self.test_data['Time'].astype(str)
        self.test_data['Time'] = pd.to_datetime(
            self.test_data['Date'] + ' ' + self.test_data['Time'],
            format='%d/%m/%Y %I:%M:%S.%f %p'
        )

    def _process_labels(self) -> None:
        """Process attack labels and create label matrix."""
        # Initialize labels DataFrame with zeros
        self.labels_data = self.test_data.copy(deep=True)
        for col in self.test_data.columns[3:]:
            self.labels_data[col] = 0

        # Process datetime in attack labels
        for time_col in ['Start Time', 'End Time']:
            self.attack_labels[time_col] = self.attack_labels[time_col].astype(str)
            self.attack_labels[time_col] = pd.to_datetime(
                self.attack_labels['Date'] + ' ' + self.attack_labels[time_col],
                format='%d/%m/%Y %H:%M:%S'
            )

        # Mark attack periods in labels
        for _, row in self.attack_labels.iterrows():
            # Find affected columns
            affected_sensors = row['Affected'].split(', ')
            matched_columns = []

            for col in self.test_data.columns[3:]:
                for sensor in affected_sensors:
                    if sensor in col:
                        matched_columns.append(col)
                        break

            # Mark attack period
            start_time = str(row['Start Time'])
            end_time = str(row['End Time'])
            mask = (self.labels_data['Time'] >= start_time) & (self.labels_data['Time'] <= end_time)
            self.labels_data.loc[mask, matched_columns] = 1

    def preprocess(self) -> None:
        """Preprocess WADI dataset with normalization."""
        logger.info("Preprocessing WADI data...")

        if self.train_data is None or self.test_data is None:
            logger.error("Data not loaded yet, please call load_data() first")
            return

        # Convert to numpy arrays
        train = self._convert_to_numpy(self.train_data)
        test = self._convert_to_numpy(self.test_data)
        labels = self._convert_to_numpy(self.labels_data)

        self.train = train
        self.test = test
        self.labels = labels

        logger.info("WADI data preprocessing completed")

    def visualize(self) -> List[Figure]:
        """Visualize the time series data."""
        figures = []

        if self.train is None or self.test is None:
            logger.error("Data not processed yet")
            return figures

        # Get feature names
        column_names = self.test_data.columns[3:].tolist()

        # Create visualization for training data
        figures.append(LinePlot.plot_time_series(
            self.train,
            column_names=column_names,
            title='WADI Training Data'
        ))

        # Create visualization for test data with anomaly labels
        figures.append(LinePlot.plot_time_series(
            self.test,
            column_names=column_names,
            title='WADI Test Data',
            labels_data=self.labels
        ))

        return figures

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.train is None or self.test is None:
            logger.error("Data not processed yet")
            return {}

        stats = {
            'train_samples': len(self.train),
            'test_samples': len(self.test),
            'dimensions': self.train.shape[1],
            'anomaly_rate': float(np.mean(np.any(self.labels == 1, axis=1))),
            'anomaly_samples': int(np.sum(np.any(self.labels == 1, axis=1))),
            'total_attacks': len(self.attack_labels)
        }

        return stats
