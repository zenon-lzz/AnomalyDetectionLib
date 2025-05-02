"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Threshold Calculation Methods
    This module provides various methods for calculating
    anomaly detection thresholds from score distributions.
==================================================
"""
from typing import Any

import numpy as np
from numpy import floating
from scipy import stats
from sklearn.metrics import precision_recall_curve


def percentile_threshold(scores: np.ndarray, percentile: float = 95) -> floating[Any]:
    """
    Calculate anomaly detection threshold using percentile method
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        percentile (float): Percentile value (default: 95)
        
    Returns:
        float: Calculated threshold value
    """
    return np.percentile(scores, percentile)


def std_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
    """
    Calculate threshold using mean plus standard deviation method
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        n_sigma (float): Number of standard deviations (default: 3)
        
    Returns:
        float: Calculated threshold value
    """
    mean = np.mean(scores)
    std = np.std(scores)
    return mean + n_sigma * std


def mad_threshold(scores: np.ndarray, n_sigma: float = 3.0) -> float:
    """
    Calculate threshold using Median Absolute Deviation (MAD) method
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        n_sigma (float): Number of MAD units (default: 3)
        
    Returns:
        float: Calculated threshold value
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median + n_sigma * mad * 1.4826  # 1.4826 makes MAD consistent with standard deviation for normal distribution


def iqr_threshold(scores: np.ndarray, k: float = 1.5) -> float:
    """
    Calculate threshold using Interquartile Range (IQR) method
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        k (float): IQR multiplier (default: 1.5)
        
    Returns:
        float: Calculated threshold value
    """
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    return q3 + k * iqr


def gaussian_threshold(scores: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate threshold by fitting Gaussian distribution
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        confidence (float): Confidence level (default: 0.95)
        
    Returns:
        float: Calculated threshold value
    """
    mean = np.mean(scores)
    std = np.std(scores)
    return stats.norm.ppf(confidence, mean, std)


def get_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the optimal threshold and corresponding metrics based on F1 score.
    
    This function computes the best threshold that maximizes the F1 score, along with
    the corresponding precision and recall values at that threshold.
    
    Args:
        scores (np.ndarray): 
            Predicted anomaly scores (higher values indicate higher likelihood of anomaly)
        labels (np.ndarray): 
            Ground truth labels (0=normal, 1=anomaly)
            
    Returns:
        best_threshold (float): Threshold value that maximizes F1 score
    """
    precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-5)  # Add small epsilon to avoid division by zero

    # Find index of maximum F1 score
    max_f1_idx = np.argmax(f1_scores)

    best_threshold = thresholds[max_f1_idx]
    
    return best_threshold
