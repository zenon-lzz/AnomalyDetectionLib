"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-12
@Description：Data standardization and scaling
==================================================
"""
import numpy as np


def minmax_scaler_column_wise(a, min_a=None, max_a=None):
    """
    Column-wise min-max normalization that scales each feature independently.
    
    This function applies min-max scaling separately to each column of the input array,
    making it suitable for datasets where features have different ranges. The transformation
    formula is: x_scaled = (x - min) / (max - min + epsilon)

    Args:
        a (numpy.ndarray): Input array to be scaled
        min_a (numpy.ndarray, optional): Minimum values for scaling. If None, computed from input array. Defaults to None.
        max_a (numpy.ndarray, optional): Maximum values for scaling. If None, computed from input array. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Scaled array with values normalized between 0 and 1
            - numpy.ndarray: Minimum values used for scaling
            - numpy.ndarray: Maximum values used for scaling

    Note:
        A small epsilon (0.0001) is added to denominator to avoid division by zero.
        This function performs column-wise normalization using np.min/max with axis=0.
    """
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a


def minmax_scaler_global(a, min_a=None, max_a=None):
    """
    Global min-max normalization that scales the entire array using a single min/max pair.
    
    This function applies min-max scaling using global minimum and maximum values across
    the entire array, preserving the relative relationships between all values. The transformation
    formula is: x_scaled = (x - min) / (max - min)

    Args:
        a (numpy.ndarray): Input array to be scaled
        min_a (float, optional): Global minimum value for scaling. If None, computed from input array. Defaults to None.
        max_a (float, optional): Global maximum value for scaling. If None, computed from input array. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Scaled array with values normalized between 0 and 1
            - float: Global minimum value used for scaling
            - float: Global maximum value used for scaling

    Note:
        Different from minmax_scaler:
        1. Uses global min/max values across entire array instead of column-wise
        2. No epsilon added to denominator (less numerically stable)
        3. Uses Python's built-in min/max instead of numpy's (slower for large arrays)
    """
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a
