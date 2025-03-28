"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description: Attention Mechanism Layer Implementation
    This module implements various attention mechanism calculation methods, including:
    1. Self-Attention
    2. Multi-Head Attention
    
    These attention mechanisms are encapsulated as PyTorch layers that can be
    directly called by other model components. They are primarily used to capture
    dependencies across different dimensions in time series anomaly detection models.
==================================================
"""
from torch import nn

from tsadlib.utils.logger import logger


class AttentionLayer(nn.Module):
    """Attention Layer in Encoder
    

    """

    def __init__(self, window_size, d_model, n_heads, d_keys=None, d_values=None,
                 mask_flag=False, scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()

        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.mask_flag = mask_flag

        if d_model % n_heads == 0:
            logger.error('d_model must be divisible by n_heads')
            raise RuntimeError('d_model must be divisible by n_heads')

        self.multiple_head_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # Set to True for [batch, seq, feature] input format
        )

        # Optional projection layer if needed
        self.out_proj = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the attention layer
        
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, d_model]
        """

        output, _ = self.multiple_head_attention(
            query=x,
            key=x,
            value=x
        )

        # Apply final projection if needed
        return self.out_proj(output)
