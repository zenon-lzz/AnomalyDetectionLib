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


class OneEarlyStopping:
    """
    Early stopping handler that monitors a single validation loss.
    
    This class implements the early stopping mechanism to prevent overfitting
    during neural network training. It tracks validation loss and stops training
    when no improvement is observed for a specified number of epochs.
    
    Attributes:
        patience (int): Number of epochs to wait after last improvement
        root_path (str): Directory to save model checkpoints
        file_name (str): Name of the saved model file
        verbose (bool): Whether to print progress messages
        counter (int): Counter for epochs without improvement
        best_score (float): Best validation score seen so far
        early_stop (bool): Flag indicating whether to stop training
        loss_min (float): Minimum validation loss observed
        delta (float): Minimum change in loss to qualify as improvement
    """

    def __init__(self, patience=10, root_path='checkpoints', file_name='model', verbose=True, delta=0):
        """
        Initialize the early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            root_path (str): Directory to save model checkpoints
            file_name (str): Name of the saved model file
            verbose (bool): Whether to print progress messages
            delta (float): Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.root_path = root_path
        self.file_name = file_name
        self.verbose = verbose
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = None  # Best validation score seen so far
        self.early_stop = False  # Signal for stopping training
        self.loss_min = np.inf  # Minimum validation loss
        self.delta = delta  # Minimum improvement threshold

    def __call__(self, loss, model):
        """
        Check if training should be stopped based on validation loss.
        
        This method is called after each epoch to evaluate the validation loss
        and determine whether to save the model or increment the early stopping counter.
        
        Args:
            loss (float): Current validation loss
            model (torch.nn.Module): Model to save if validation loss improves
            
        Returns:
            None
        """
        score = -loss
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score + self.delta:
            # Score didn't improve enough
            self.counter += 1
            if self.verbose:
                logger.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Score improved
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        """
        Save model checkpoint when validation loss decreases.
        
        Args:
            loss (float): Current validation loss
            model (torch.nn.Module): Model to save
            
        Returns:
            None
        """
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')

        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        torch.save(model.state_dict(), os.path.join(self.root_path, f'{self.file_name}.pth'))
        self.loss_min = loss


class TwoEarlyStopping:
    """
    Early stopping handler that monitors two validation losses.
    
    This class extends the early stopping concept to monitor two different
    validation losses simultaneously. Training stops when neither loss shows
    improvement for a specified number of epochs.
    
    Attributes:
        patience (int): Number of epochs to wait after last improvement
        root_path (str): Directory to save model checkpoints
        file_name (str): Name of the saved model file
        verbose (bool): Whether to print progress messages
        counter (int): Counter for epochs without improvement
        best_score (float): Best primary validation score seen so far
        best_score2 (float): Best secondary validation score seen so far
        early_stop (bool): Flag indicating whether to stop training
        loss_min (float): Minimum primary validation loss observed
        loss2_min (float): Minimum secondary validation loss observed
        delta (float): Minimum change in loss to qualify as improvement
    """

    def __init__(self, patience=10, root_path='checkpoints', file_name='model', verbose=True, delta=0):
        """
        Initialize the dual-loss early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            root_path (str): Directory to save model checkpoints
            file_name (str): Name of the saved model file
            verbose (bool): Whether to print progress messages
            delta (float): Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.root_path = root_path
        self.file_name = file_name
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.loss_min = np.inf
        self.loss2_min = np.inf
        self.delta = delta

    def __call__(self, loss, loss2, model):
        """
        Check if training should be stopped based on two validation losses.
        
        This method is called after each epoch to evaluate both validation losses
        and determine whether to save the model or increment the early stopping counter.
        Training continues if either loss improves.
        
        Args:
            loss (float): Current primary validation loss
            loss2 (float): Current secondary validation loss
            model (torch.nn.Module): Model to save if validation losses improve
            
        Returns:
            None
        """
        score = -loss
        score2 = -loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(loss, loss2, model)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.verbose:
                logger.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(loss, loss2, model)
            self.counter = 0

    def save_checkpoint(self, loss, loss2, model):
        """
        Save model checkpoint when validation losses decrease.
        
        Args:
            loss (float): Current primary validation loss
            loss2 (float): Current secondary validation loss
            model (torch.nn.Module): Model to save
            
        Returns:
            None
        """
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        torch.save(model.state_dict(), os.path.join(self.root_path, f'{self.file_name}.pth'))
        self.loss_min = loss
        self.loss2_min = loss2
