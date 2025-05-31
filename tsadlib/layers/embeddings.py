"""
=================================================
@Author: Zenon
@Date: 2025-03-16
@Description: Various embedding implementations for time series data procession
==================================================
"""
import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """Standard positional encoding using sine and cosine functions.
    
    Used to give the model information about the position of each element in the sequence.
    Implementation follows the original Transformer paper's positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length to pre-compute
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # calculate :math: \frac{pos}{10000^[\frac{pos}{d_{model}}}}
        # position: numerator
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # div_term: reciprocal denominator
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        # If d_model is odd, the number of columns for cosine will be one less than for sine
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:, :-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # When the batch dimension is added, it can be automatically extended to any batch_size through PyTorch's broadcast mechanism
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Value embedding using 1D convolution.
    
    Transforms raw input values into a higher dimensional space using Conv1d.
    Provides local context through convolution operation.
    """

    def __init__(self, input_channels, d_model):
        """
        Args:
            input_channels: Number of input channels/features
            d_model: Output embedding dimension
        """
        super().__init__()
        # In PyTorch 1.5.0, the padding calculation for Conv1d was updated
        # Before 1.5.0: padding = (kernel_size - 1) // 2 = 2
        # After 1.5.0: padding = kernel_size // 2 = 1
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=input_channels, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """Combines value, position, and temporal embeddings.
    
    Complete embedding module that combines:
    1. Value embedding (TokenEmbedding)
    2. Positional encoding (PositionalEmbedding)
    3. Temporal embedding (TemporalEmbedding or TimeFeatureEmbedding)
    
    Application:
    - Main embedding layer for time series models
    - When both value patterns and temporal patterns are important
    """

    def __init__(self, input_channels, d_model, dropout=0.1):
        """
        Args:
            input_channels: Number of input features
            d_model: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.value_embedding = TokenEmbedding(input_channels=input_channels, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: Input values [batch_size, window_size, features]
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
