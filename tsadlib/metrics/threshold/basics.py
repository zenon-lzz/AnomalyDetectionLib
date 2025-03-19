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
from sklearn.metrics import f1_score


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


def find_best_threshold_by_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the best threshold that maximizes F1 score
    
    Args:
        scores (np.ndarray): Array of anomaly scores
        labels (np.ndarray): Array of true labels
        
    Returns:
        float: Threshold value that maximizes F1 score
    """
    # 对分数进行排序，同时记录索引以保持与标签的对应关系
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]

    best_f1 = 0
    best_threshold = sorted_scores[0]
    prev_score = None

    # 遍历排序后的分数
    for i, score in enumerate(sorted_scores):
        # 如果当前分数与前一个分数相同，跳过
        if score == prev_score:
            continue

        # 使用当前分数作为阈值
        threshold = score
        pred_labels = (scores > threshold).astype(int)
        current_f1 = f1_score(labels, pred_labels, average='binary')

        # 更新最佳阈值
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

        prev_score = score

    return best_threshold
