"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description: This file defines the abstract base class for anomaly detection experiments, providing the basic structure for model, data, and experiment workflow management.
==================================================
"""
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from tsadlib import log
from tsadlib.configs.type import ConfigType
from tsadlib.models import *


class ExperimentBase(ABC):
    """
    Abstract base class for anomaly detection experiments.
    This class provides the basic structure for managing models, data loading, device selection, and experiment workflow.
    Subclasses should implement the abstract methods to define specific experiment logic.
    """

    def __init__(self, args: ConfigType):
        """
        Initialize the experiment with configuration arguments.
        Sets up the model dictionary and selects the computation device.
        
        Args:
            args (ConfigType): Configuration object containing experiment parameters.
        """
        self.args = args
        # Mapping of model names to their corresponding classes
        self.model_dict = {
            'TimesNet': TimesNet,
            'MtsCID': MtsCID,
            'MEMTO': MEMTO
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.metric_record_flag = False

    def _acquire_device(self) -> torch.device:
        """
        Select and acquire the computation device (GPU or CPU) based on configuration.
        Sets CUDA_VISIBLE_DEVICES if using CUDA GPU.
        
        Returns:
            torch.device: The selected device for computation.
        """
        if self.args.use_gpu:
            if self.args.gpu_type == 'cuda':
                device = torch.device(f'cuda:{self.args.gpu}')
                log.info(f'Use GPU: cuda:{self.args.gpu}')
            elif self.args.gpu_type == 'mps':
                device = torch.device('mps')
                log.info('Use GPU: mps')
            else:
                device = torch.device('cpu')
                log.warning(f'{self.args.gpu_type} is not supported.')
        else:
            device = torch.device('cpu')
            log.info('Use CPU to train')
        return device

    @abstractmethod
    def _build_model(self):
        """
        Abstract method for building the model.
        Should be implemented by subclasses to construct and return the model instance.
        """
        pass

    @abstractmethod
    def _get_data(self, split_way: str):
        """
        Abstract method for loading data.
        Should be implemented by subclasses to load and return datasets and dataloaders.
        """
        pass

    @abstractmethod
    def train(self, setting: str):
        """
        Abstract method for training the model.
        Should be implemented by subclasses to define the training process.
        """
        pass

    @abstractmethod
    def validate(self, data_loader: DataLoader):
        """
        Abstract method for validating the model.
        Should be implemented by subclasses to define the validation process.
        """
        pass

    @abstractmethod
    def test(self, setting: str):
        """
        Abstract method for testing the model.
        Should be implemented by subclasses to define the testing process.
        """
        pass
