"""
=================================================
@Author: Zenon
@Date: 2025-03-29
@Description: Clustering Utilities Module
    This module implements various clustering algorithms for time series anomaly detection.
==================================================
"""
import time

import torch
from kmeans_pytorch import kmeans

from tsadlib.utils.logger import logger


def k_means_clustering(x, num_clusters, d_model, device='cuda', tol=1e-3):
    """
    GPU-accelerated K-Means clustering implementation.
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, ...] to be clustered
        num_clusters (int): Number of clusters to form
        d_model (int): Feature dimension for reshaping
        device (str): Device to run on ('cuda' or 'cpu')
        tol (float): Relative tolerance threshold for convergence (default: 1e-4)
        
    Returns:
        torch.Tensor: Cluster centers of shape [num_clusters, d_model]
    """
    start = time.time()
    # Reshape input to 2D tensor [N, d_model]
    x = x.view([-1, d_model])
    logger.info(f'Running K-Means Clustering with {num_clusters} clusters')

    # Perform K-Means clustering
    _, cluster_centers = kmeans(
        X=x,
        num_clusters=num_clusters,
        distance='euclidean',
        device=device,
        tol=tol
    )

    logger.info('K-Means clustering completed in {:.2f}s'.format(time.time() - start))
    return cluster_centers
