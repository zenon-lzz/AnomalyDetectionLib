"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-12
@Description：Data standardization and scaling
==================================================
"""
import numpy as np


def minmax_scaler(a, min_a=None, max_a=None):
    """
    Performs min-max scaling on input array to normalize values between 0 and 1.

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
    """
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a
