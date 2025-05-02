"""
=================================================
@Author: Zenon
@Date: 2025-05-02
@Description: TensorBoard Utilities Module
    This module provides utility functions for logging to TensorBoard during model training.
==================================================
"""
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    A wrapper class for TensorBoard logging functionality.
    
    This class provides convenient methods for logging various types of data
    (scalars, histograms, images, etc.) to TensorBoard during model training.
    """

    def __init__(self, log_dir: str = 'runs', comment: str = ''):
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.
            comment (str): Optional comment to append to the log directory name.
        """
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def log_scalar(self, tag: str, value: Union[float, torch.Tensor, np.ndarray],
                   step: int, walltime: Optional[float] = None) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag (str): Data identifier
            value (float|Tensor|ndarray): Value to log
            step (int): Global step value
            walltime (float, optional): Optional override for default walltime
            
        Returns:
            None
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()

        self.writer.add_scalar(tag, value, step, walltime)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, Union[float, torch.Tensor, np.ndarray]],
                    step: int, walltime: Optional[float] = None) -> None:
        """
        Log multiple scalar values to TensorBoard under the same main tag.
        
        Args:
            main_tag (str): Parent name for the tags
            tag_scalar_dict (dict): Key-value pairs storing tag and corresponding values
            step (int): Global step value
            walltime (float, optional): Optional override for default walltime
            
        Returns:
            None
        """
        scalar_dict = {}
        for tag, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.item()
            scalar_dict[tag] = value

        self.writer.add_scalars(main_tag, scalar_dict, step, walltime)

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray],
                      step: int, bins: str = 'tensorflow', walltime: Optional[float] = None) -> None:
        """
        Log a histogram of values to TensorBoard.
        
        Args:
            tag (str): Data identifier
            values (Tensor|ndarray): Values to build histogram
            step (int): Global step value
            bins (str): One of {'tensorflow','auto', 'fd', ...}
            walltime (float, optional): Optional override for default walltime
            
        Returns:
            None
        """
        self.writer.add_histogram(tag, values, step, bins, walltime)

    def log_image(self, tag: str, img_tensor: Union[torch.Tensor, np.ndarray],
                  step: int, walltime: Optional[float] = None,
                  dataformats: str = 'CHW') -> None:
        """
        Log an image to TensorBoard.
        
        Args:
            tag (str): Data identifier
            img_tensor (Tensor|ndarray): Image data
            step (int): Global step value
            walltime (float, optional): Optional override for default walltime
            dataformats (str): Image data format specification
            
        Returns:
            None
        """
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats, walltime=walltime)

    def log_model_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor = None) -> None:
        """
        Log the model's computational graph to TensorBoard.
        
        Args:
            model (nn.Module): Model to log
            input_to_model (Tensor, optional): Example input for tracing
            
        Returns:
            None
        """
        self.writer.add_graph(model, input_to_model)

    def close(self) -> None:
        """
        Close the TensorBoard writer and flush any remaining data.
        
        Returns:
            None
        """
        self.writer.close()


def create_tensorboard_logger(log_dir: str = 'runs', comment: str = '') -> TensorBoardLogger:
    """
    Create and return a TensorBoard logger instance.
    
    Args:
        log_dir (str): Directory where TensorBoard logs will be saved
        comment (str): Optional comment to append to the log directory name
        
    Returns:
        TensorBoardLogger: Initialized logger instance
    """
    return TensorBoardLogger(log_dir=log_dir, comment=comment)
