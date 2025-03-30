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


class FixedEmbedding(nn.Module):
    """Fixed positional encoding implemented as non-trainable embedding.
    
    Similar to PositionalEmbedding but wrapped as nn.Embedding with fixed weights.
    Primarily used for encoding temporal features with fixed patterns.
    """

    def __init__(self, input_channels, d_model):
        """
        Args:
            input_channels: Input vocabulary size (e.g., 24 for hours)
            d_model: Output embedding dimension
        """
        super().__init__()

        w = torch.zeros(input_channels, d_model).float()
        w.require_grad = False

        position = torch.arange(0, input_channels).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(input_channels, d_model)
        # use the positional encoding as embedding's weight and not update it
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Comprehensive time feature embedding.
    
    Combines multiple temporal embeddings (month, day, weekday, hour, minute)
    Can use either fixed or trainable embeddings based on embedding_type.
    
    Application:
    - When timestamp information is available
    - When temporal patterns are important (e.g., daily, weekly patterns)
    """

    def __init__(self, d_model, embedding_type='fixed', freq='h'):
        """
        Args:
            d_model: Output embedding dimension
            embedding_type: 'fixed' for FixedEmbedding or 'learned' for trainable embedding
            freq: Time frequency ('h' for hourly, 't' for minutely)
        """
        super().__init__()

        # Minute divisions: 4 intervals of 15 minutes each (0-14, 15-29, 30-44, 45-59)
        minute_size = 4

        # Hours in a day (0-23)
        hour_size = 24

        # Days in a week (0-6)
        weekday_size = 7

        # Days in a month (1-31) plus padding/special token (0)
        day_size = 32

        # Months in a year (1-12) plus padding/special token (0)
        month_size = 13

        Embed = FixedEmbedding if embedding_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection for time features.
    
    Simpler alternative to TemporalEmbedding using linear transformation.
    
    Application:
    - When time features are already preprocessed
    - When computational efficiency is priority
    """

    def __init__(self, d_model, embedding_type='timeF', freq='h'):
        """
        Args:
            d_model: Output embedding dimension
            embedding_type: Type of embedding (only 'timeF' supported)
            freq: Time frequency determining input dimension
        """
        super().__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


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

    def __init__(self, input_channels, d_model, embedding_type='normal', freq='h', dropout=0.1):
        """
        Args:
            input_channels: Number of input features
            d_model: Output embedding dimension
            embedding_type: Type of temporal embedding ('fixed', 'normal', or 'timeF')
            freq: Time frequency
            dropout: Dropout rate
        """
        super().__init__()

        self.value_embedding = TokenEmbedding(input_channels=input_channels, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embedding_type == 'fixed':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embedding_type=embedding_type, freq=freq)
        elif embedding_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embedding_type=embedding_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        """
        Args:
            x: Input values [batch_size, window_size, features]
            x_mark: Temporal features [batch_size, window_size, time_features]
                   If None, only value and position embeddings are used
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
