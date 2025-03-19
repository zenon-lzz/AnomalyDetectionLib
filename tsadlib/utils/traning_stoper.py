"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Early Stopping Implementation
    This module provides early stopping functionality for neural network training.
    It monitors validation loss and stops training when no improvement is seen
    over a specified number of epochs.
==================================================
"""
import os.path

import numpy as np
import torch

from tsadlib import logger


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs (patience).
    
    Features:
    - Saves best model checkpoint
    - Provides stopping signal to training loop
    - Supports minimum improvement threshold
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait before stopping (default: 7)
            verbose (bool): Whether to print improvement messages (default: False)
            delta (float): Minimum change in loss to qualify as improvement (default: 0)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = None  # Best validation score seen so far
        self.early_stop = False  # Signal for stopping training
        self.val_loss_min = np.inf  # Minimum validation loss
        self.delta = delta  # Minimum improvement threshold

    def __call__(self, val_loss, model, path: str = '', file_name: str | None = None):
        """
        Check for early stopping conditions and save checkpoint if needed.
        
        Args:
            val_loss (float): Current validation loss
            model (torch.nn.Module): Model to save if improved
            path (str): Directory to save model checkpoint
        
        Returns:
            bool: True if training should stop, False otherwise
        
        Note:
            Score is negative of loss (higher is better)
            Counter increases when score doesn't improve by at least delta
        """
        score = -val_loss
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, file_name)
        elif score < self.best_score + self.delta:
            # Score didn't improve enough
            self.counter += 1
            logger.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Score improved
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, file_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, file_name: str | None):
        """
        Save model checkpoint when validation loss improves.
        
        Args:
            val_loss (float): Current validation loss
            model (torch.nn.Module): Model state to save
            path (str): Directory to save checkpoint
            file_name (None | str): save checkpoint's file name
        """
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, f'{'model' if file_name is None else file_name}.pth'))
        self.val_loss_min = val_loss
