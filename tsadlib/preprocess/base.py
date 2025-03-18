"""
=================================================
@Author: Zenon
@Date: 2025-03-12
@Description：Dataset base classes that define common interfaces and methods
==================================================
"""
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np

from tsadlib import logger


class BaseDataset(ABC):
    """Base dataset class that defines common interfaces and methods for data processing.
    
    This abstract class provides a foundation for implementing dataset-specific processing
    with common functionality like loading, preprocessing, saving and retrieving data.
    """

    def __init__(self,
                 data_dir: str,
                 save_dir: str = None):
        """Initialize the base dataset.
        
        Args:
            data_dir: Directory containing the raw data files
            save_dir: Directory to save processed data files. If None, saving is disabled.
        """
        self.data_dir = data_dir
        self.save_dir = save_dir

        # Dataset container
        self.train = None
        self.test = None
        self.labels = None

        # If the save_dir is not None, a directory is created.
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    @abstractmethod
    def load_data(self) -> None:
        """Load the raw data from data_dir.
        
        This method should be implemented by child classes to handle dataset-specific loading.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess the raw data.
        
        This method should be implemented by child classes to handle dataset-specific preprocessing.
        """
        pass

    def save(self, ) -> None:
        """Save the processed data to numpy files in save_dir."""
        if self.save_dir is None:
            logger.error("The save_dir is not specified, operation is skipped.")
            return

        self._save_as_npy()

        logger.info(f"Data has been saved to {self.save_dir}")

    def _save_as_npy(self) -> None:
        """Internal method to save data as numpy files."""
        np.save(os.path.join(self.save_dir, 'train.npy'), self.train)
        np.save(os.path.join(self.save_dir, 'labels.npy'), self.labels)
        np.save(os.path.join(self.save_dir, 'test.npy'), self.test)

    def get_data(self) -> Tuple:
        """Get the processed dataset.

        Returns:
            tuple: A tuple containing (train, labels, test) arrays
        """
        return self.train, self.labels, self.test

    def visualize(self) -> None:
        """Visualize the dataset.
        
        This method can be implemented by child classes for dataset-specific visualization.
        """
        logger.warning("Visualization not implemented in base class. Please implement in child class.")

    def get_statistics(self) -> Dict:
        """Get statistical information about the dataset.
        
        This method can be implemented by child classes for dataset-specific statistics.

        Returns:
            dict: An empty dictionary in base class
        """
        logger.warning("Statistics not implemented in base class. Please implement in child class.")
        return {}
