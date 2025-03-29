"""
=================================================
@Author: Zenon
@Date: 2025-03-29
@Description: Custom Loss Functions Module
    This module implements various custom loss functions for time series anomaly detection.
==================================================
"""
import torch
import torch.nn.functional as F
from torch import nn


class EntropyLoss(nn.Module):
    """
    Entropy-based loss function for measuring the uncertainty or information content.
    
    This loss is particularly useful for:
    - Encouraging diversity in output distributions
    - Preventing model overconfidence
    - Regularizing probabilistic outputs
    
    Args:
        eps (float): Small constant for numerical stability (default: 1e-12)
    """

    def __init__(self, eps=1e-12):
        """
        Initialize the entropy loss with numerical stability constant.
        
        Args:
            eps (float): Small value to prevent log(0) (default: 1e-12)
        """
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        Compute the entropy loss for input probabilities.
        
        Args:
            x (torch.Tensor): Input probability tensor of shape [B, ..., C]
                             where values should be in [0,1] range
                             
        Returns:
            torch.Tensor: Scalar entropy loss value
        """
        # Calculate element-wise entropy: -p*log(p)
        loss = -1 * x * torch.log(x + self.eps)
        # Sum over the last dimension (typically classes/channels)
        loss = torch.sum(loss, dim=-1)
        # Average over batch and other dimensions
        loss = torch.mean(loss)
        return loss


class GatheringLoss(nn.Module):
    """A loss function that measures the similarity between queries and memory items.
    
    This loss encourages queries to be close to their most similar memory items in the 
    feature space. It's commonly used in memory-augmented neural networks.
    
    Args:
        reduce (bool): If True, returns scalar loss. If False, returns per-element loss.
    """

    def __init__(self, reduce=True):
        """Initialize the GatheringLoss.
        
        Args:
            reduce (bool): Determines reduction behavior of the loss.
        """
        super().__init__()
        self.reduce = reduce

    def get_score(self, query, key):
        """Compute similarity scores between queries and keys.
        
        Args:
            query (torch.Tensor): Query tensor of shape (T, C)
            key (torch.Tensor): Key tensor of shape (M, C)
            
        Returns:
            torch.Tensor: Similarity scores of shape (T, M)
        """
        # Compute dot product similarity between queries and keys
        score = torch.matmul(query, torch.t(key))  # (T, C) x (C, M) -> (T, M)
        # Normalize scores to probabilities using softmax
        score = F.softmax(score, dim=1)  # (T, M)
        return score

    def forward(self, queries, items):
        """Compute the gathering loss between queries and memory items.
        
        Args:
            queries (torch.Tensor): Input queries of shape (N, L, C)
            items (torch.Tensor): Memory items of shape (M, C)
            
        Returns:
            torch.Tensor: The computed gathering loss
        """
        # Get batch size and feature dimension
        batch_size = queries.size(0)
        d_model = queries.size(-1)

        # Initialize MSE loss with appropriate reduction
        loss_mse = torch.nn.MSELoss(reduction='mean' if self.reduce else 'none')

        # Reshape queries to (T, C) where T = N*L
        queries = queries.contiguous().view(-1, d_model)  # (N*L, C)

        # Get similarity scores between queries and items
        score = self.get_score(queries, items)  # (T, M)

        # Find most similar item for each query
        _, indices = torch.topk(score, 1, dim=1)  # (T, 1)

        # Compute MSE between queries and their most similar items
        gathering_loss = loss_mse(queries, items[indices].squeeze(1))

        # Return reduced loss if specified
        if self.reduce:
            return gathering_loss

        # For non-reduced case: sum over features and reshape to (N, L)
        gathering_loss = torch.sum(gathering_loss, dim=-1)  # (T,)
        gathering_loss = gathering_loss.contiguous().view(batch_size, -1)  # (N, L)

        return gathering_loss
