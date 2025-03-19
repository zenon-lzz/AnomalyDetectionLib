"""
=================================================
@Author: Zenon
@Date: 2025-03-19
@Description：Anomaly Detection Point Adjustment Strategies
    This module provides various methods for adjusting
    anomaly detection points to improve detection accuracy.
==================================================
"""
from typing import Tuple

import numpy as np


def point_adjustment(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust anomaly detection results by expanding detected anomaly points to cover the entire anomaly interval
    
    Args:
        gt (np.ndarray): Ground truth label sequence
        pred (np.ndarray): Predicted label sequence
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted ground truth and prediction labels
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Expand backwards
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Expand forwards
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def window_adjustment(gt: np.ndarray, pred: np.ndarray, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust anomaly detection results using sliding window approach
    
    Args:
        gt (np.ndarray): Ground truth label sequence
        pred (np.ndarray): Predicted label sequence
        window_size (int): Size of sliding window
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted ground truth and prediction labels
    """
    adjusted_pred = pred.copy()
    half_window = window_size // 2

    for i in range(len(gt)):
        if pred[i] == 1:
            # Get window range
            start = max(0, i - half_window)
            end = min(len(gt), i + half_window + 1)

            # If true anomaly exists in window, mark entire window as anomaly
            if np.any(gt[start:end] == 1):
                adjusted_pred[start:end] = 1

    return gt, adjusted_pred


def threshold_adjustment(scores: np.ndarray, threshold: float,
                         adjust_ratio: float = 0.1) -> Tuple[float, np.ndarray]:
    """
    Dynamically adjust threshold based on anomaly score distribution
    
    Args:
        scores (np.ndarray): Anomaly score sequence
        threshold (float): Initial threshold
        adjust_ratio (float): Threshold adjustment ratio
        
    Returns:
        Tuple[float, np.ndarray]: Adjusted threshold and prediction labels
    """
    score_mean = np.mean(scores)
    score_std = np.std(scores)

    # Adjust threshold based on score distribution
    if score_mean > threshold:
        adjusted_threshold = threshold * (1 + adjust_ratio)
    else:
        adjusted_threshold = threshold * (1 - adjust_ratio)

    # Generate prediction labels
    predictions = (scores > adjusted_threshold).astype(int)

    return adjusted_threshold, predictions


def density_adjustment(gt: np.ndarray, pred: np.ndarray,
                       density_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust predictions based on anomaly point density
    
    Args:
        gt (np.ndarray): Ground truth label sequence
        pred (np.ndarray): Predicted label sequence
        density_threshold (float): Density threshold for adjustment
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted ground truth and prediction labels
    """
    adjusted_pred = pred.copy()
    window_size = 10

    for i in range(len(gt) - window_size + 1):
        window = pred[i:i + window_size]
        density = np.sum(window) / window_size

        if density > density_threshold:
            adjusted_pred[i:i + window_size] = 1

    return gt, adjusted_pred
