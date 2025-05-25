"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description:
==================================================
"""
"""
Early stopping utilities for neural network training.

This module provides implementations of early stopping mechanisms that monitor
validation metrics during model training and stop training when no improvement
is observed over a specified number of epochs.
"""
import os.path

import numpy as np
import torch

from tsadlib import log, EarlyStoppingModeEnum


class OneEarlyStopping:
    """Early stopping handler that monitors a single validation metric.
    
    This class implements early stopping mechanism to prevent overfitting during
    neural network training. It tracks a single metric (e.g. loss or accuracy)
    and stops training when no improvement is observed for a specified number
    of epochs.
    
    Attributes:
        patience (int): Number of epochs to wait after last improvement
        root_path (str): Directory to save model checkpoints
        file_name (str): Name of the saved model file
        verbose (bool): Whether to print progress messages
        counter (int): Counter for epochs without improvement
        best_score (float): Best validation score seen so far
        early_stop (bool): Flag indicating whether to stop training
        optimal_value (float): Optimal validation value observed
        delta (float): Minimum change in metric to qualify as improvement
        mode (EarlyStoppingModeEnum): Whether to minimize or maximize the metric
    """

    def __init__(self, patience=10, root_path='checkpoints', file_name='model',
                 verbose=True, delta=0, mode: EarlyStoppingModeEnum = EarlyStoppingModeEnum.MINIMIZE):
        """Initialize the early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            root_path (str): Directory to save model checkpoints
            file_name (str): Name of the saved model file
            verbose (bool): Whether to print progress messages
            delta (float): Minimum change in metric to qualify as improvement
            mode (EarlyStoppingModeEnum): Whether to minimize or maximize the metric
        """
        self.patience = patience
        self.root_path = root_path
        self.file_name = file_name
        self.verbose = verbose
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = None  # Best validation score seen so far
        self.early_stop = False  # Signal for stopping training
        self.delta = delta  # Minimum improvement threshold
        self.mode = mode  # Early stopping mode
        # Optimal validation value
        if mode == EarlyStoppingModeEnum.MINIMIZE:
            self.optimal_value = np.inf
        else:
            self.optimal_value = -np.inf

    def __call__(self, metric, model) -> bool:
        """Check if training should be stopped based on validation metric.
        
        Args:
            metric (float): Current metric value to evaluate
            model (torch.nn.Module): Model to save if metric improves
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if self.mode == EarlyStoppingModeEnum.MINIMIZE:
            score = -metric
        else:
            score = metric
            
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            # Score didn't improve enough
            self.counter += 1
            if self.verbose:
                log.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Score improved
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, metric, model):
        """Save model checkpoint when validation metric improves.
        
        Args:
            metric (float): Current metric value
            model (torch.nn.Module): Model to save
        """
        if self.verbose:
            log.info(
                f'Validation Metric {"decreased" if self.mode == EarlyStoppingModeEnum.MINIMIZE else "improved"} ({self.optimal_value:.6f} --> {metric:.6f}).  Saving model ...')

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        torch.save(model.state_dict(), os.path.join(self.root_path, f'{self.file_name}.pth'))
        self.optimal_value = metric


class TwoEarlyStopping:
    """Early stopping handler that monitors two validation metrics.
    
    This class extends the early stopping concept to monitor two different
    metrics simultaneously (e.g. loss and accuracy). Training stops when neither
    metric shows improvement for the specified number of epochs.
    """

    def __init__(self, patience=10, root_path='checkpoints', file_name='model',
                 verbose=True, delta=0, mode: EarlyStoppingModeEnum = EarlyStoppingModeEnum.MINIMIZE):
        self.patience = patience
        self.root_path = root_path
        self.file_name = file_name
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        if mode == EarlyStoppingModeEnum.MINIMIZE:
            self.optimal_value = np.inf
            self.optimal_value2 = np.inf
        else:
            self.optimal_value = -np.inf
            self.optimal_value2 = -np.inf

    def __call__(self, metric, metric2, model) -> bool:
        if self.mode == EarlyStoppingModeEnum.MINIMIZE:
            score = -metric
            score2 = -metric2
        else:
            score = metric
            score2 = metric2
            
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(metric, metric2, model)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.verbose:
                log.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(metric, metric2, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, metric, metric2, model):
        if self.verbose:
            log.info(
                f'Validation Metric {'decreased' if self.mode == EarlyStoppingModeEnum.MINIMIZE else 'improved'} ({self.optimal_value:.6f} --> {metric:.6f}).  Saving model ...')
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        torch.save(model.state_dict(), os.path.join(self.root_path, f'{self.file_name}.pth'))
        self.optimal_value = metric
        self.optimal_value2 = metric2
