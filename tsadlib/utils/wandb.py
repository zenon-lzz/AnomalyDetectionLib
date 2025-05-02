"""
=================================================
@Author: Zenon
@Date: 2025-05-02
@Description: Weights & Biases (wandb) Utility Functions
    This module provides utility functions for logging and monitoring with wandb.
==================================================
"""
from typing import Dict, Any, Optional

import wandb


def init_wandb(project: str,
               config: Dict[str, Any],
               name: Optional[str] = None,
               tags: Optional[list] = None) -> None:
    """
    Initialize a new wandb run with the given configuration.
    
    Args:
        project (str): Name of the wandb project
        config (Dict[str, Any]): Configuration dictionary to log
        name (Optional[str]): Name for this run (default: None)
        tags (Optional[list]): List of tags for this run (default: None)
        
    Returns:
        None
    """
    wandb.init(project=project, config=config, name=name, tags=tags)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to wandb with optional step number.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics to log
        step (Optional[int]): Step number for time series logging (default: None)
        
    Returns:
        None
    """
    if step is not None:
        metrics['step'] = step
    wandb.log(metrics)


def watch_model(model, log: str = 'gradients', log_freq: int = 100) -> None:
    """
    Watch model parameters and gradients in wandb.
    
    Args:
        model: PyTorch model to monitor
        log (str): What to log ('gradients', 'parameters', or 'all')
        log_freq (int): Logging frequency in steps (default: 100)
        
    Returns:
        None
    """
    wandb.watch(model, log=log, log_freq=log_freq)


def log_artifacts(file_path: str,
                  name: str,
                  artifact_type: str = 'model',
                  metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log files as artifacts to wandb.
    
    Args:
        file_path (str): Path to file or directory to log
        name (str): Name for the artifact
        artifact_type (str): Type of artifact ('model', 'data', etc.)
        metadata (Optional[Dict[str, Any]]): Additional metadata (default: None)
        
    Returns:
        None
    """
    artifact = wandb.Artifact(name, type=artifact_type)
    artifact.add_file(file_path)
    if metadata:
        artifact.metadata.update(metadata)
    wandb.log_artifact(artifact)


def log_confusion_matrix(y_true, y_pred, class_names: list) -> None:
    """
    Log a confusion matrix to wandb.
    
    Args:
        y_true: Array-like of true labels
        y_pred: Array-like of predicted labels
        class_names (list): List of class names for labeling
        
    Returns:
        None
    """
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )})


def log_histogram(values, name: str, step: Optional[int] = None) -> None:
    """
    Log a histogram of values to wandb.
    
    Args:
        values: Array-like of values to plot
        name (str): Name for the histogram
        step (Optional[int]): Step number for time series (default: None)
        
    Returns:
        None
    """
    wandb.log({name: wandb.Histogram(values)}, step=step)


def finish_run() -> None:
    """
    Finish the current wandb run and sync all data.
    
    Returns:
        None
    """
    wandb.finish()
