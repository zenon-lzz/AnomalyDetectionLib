"""
=================================================
@Author: Zenon
@Date: 2025-05-27
@Description：Utility functions for various random operations.
==================================================
"""
import random

import numpy as np
import torch


def set_random_seed(seed: int):
    # sets the seed for the python random module
    random.seed(seed)
    # Set PyTorch's CPU random seed
    torch.manual_seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)

    # CUDA
    if torch.cuda.is_available():
        # Set PyTorch's GPU random seed
        torch.cuda.manual_seed(seed)
        # If multiple Gpus are used, set the random number seed for all Gpus
        torch.cuda.manual_seed_all(seed)

    # mps
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
