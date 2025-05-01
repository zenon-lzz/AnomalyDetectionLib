"""
=================================================
@Author: Zenon
@Date: 2025-05-01
@Description: This module provides utility functions for GPU operations.
==================================================
"""
import torch.cuda


def empty_gpu_cache():
    """
    Clear the CUDA memory cache if a CUDA-enabled GPU is available.
    This function helps release unused GPU memory to avoid memory leaks during model training or inference.
    """
    # Check if CUDA is available before attempting to clear the cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
