"""
=================================================
@Author: Zenon
@Date: 2025-03-15
@Description：TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
Code Repository: https://github.com/thuml/TimesNet
==================================================
"""
import torch
import torch.nn.functional as F
from torch import nn

from tsadlib.configs.type import ConfigType
from tsadlib.layers.convolution_block import InceptionBlockV1
from tsadlib.layers.embeddings import DataEmbedding


def FFT_for_Period(x, k=2):
    """
    Extract top-k dominant periods from time series using Fast Fourier Transform.
    
    Process:
    1. Apply FFT to get frequency components
    2. Calculate amplitude spectrum
    3. Find dominant frequencies (excluding DC component)
    4. Convert frequencies to periods
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, T, C] where:
            B: batch size
            T: sequence length
            C: number of channels/features
        k (int): Number of top periods to extract (default: 2)
    
    Returns:
        tuple: (periods, period_weights)
            - periods: Array of k dominant periods, dimension: k
            - period_weights: Corresponding amplitudes for each period, dimension: [B, k]
    """
    # [B, T, C]
    # Calculate non-negative frequency components
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    # abs: calculate the amplitudes by the modulus of complex number
    # The B and T dimensions are averaged in turn
    frequency_list = abs(xf).mean(0).mean(-1)
    # After Fourier transform, frequency_list[0] represents the DC component (signal mean)
    # In time series analysis, we focus more on periodic changes rather than absolute values
    # Setting DC component to 0:
    # - Eliminates baseline shift effects
    # - Makes period detection more focused on dynamic changes
    # - Prevents DC component (usually large amplitude) from interfering with periodic feature extraction
    frequency_list[0] = 0
    # top_list is the indies of top k-largest numbers
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    Core block of TimesNet that converts 1D time series to 2D variations.
    
    This block performs the following key operations:
    1. Extracts dominant periods using FFT
    2. Reshapes the time series into 2D representations based on periods
    3. Applies 2D convolutions using Inception blocks
    4. Adaptively aggregates multi-period features
    
    Args:
        configs (ConfigType): Configuration object containing:
            - window_size: Length of input sequence
            - top_k: Number of top periods to consider
            - d_model: Input dimension
            - d_ff: Hidden dimension for feed-forward network
            - num_kernels: Number of kernels in Inception blocks
    """

    def __init__(self, configs: ConfigType):
        """
        Initialize TimesBlock with configuration parameters.
        
        Args:
            configs: Configuration containing:
                window_size: Input sequence length
                top_k: Number of periods to consider
                d_model: Model dimension
                d_ff: Feed-forward dimension
                num_kernels: Number of inception kernels
        """
        super(TimesBlock, self).__init__()
        self.window_size = configs.window_size
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(configs.d_model, configs.dimension_fcl,
                             num_kernels=configs.num_kernels),
            nn.GELU(),
            InceptionBlockV1(configs.dimension_fcl, configs.d_model,
                             num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        """
        Process time series through period-based 2D convolution.
        
        Steps:
        1. Extract periods using FFT
        2. For each period:
           - Reshape to 2D
           - Apply inception convolution
           - Reshape back to 1D
        3. Aggregate results with adaptive weights
        4. Add residual connection
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, N]
                B: batch size
                T: sequence length
                N: number of features
        
        Returns:
            torch.Tensor: Processed tensor of shape [B, T, N]
        """
        B, T, N = x.size()  # Extract dimensions from input tensor
        # Get dominant periods and their weights through FFT
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        # Process each period separately
        for i in range(self.k):
            period = period_list[i]
            # Handle sequences that don't divide evenly by the period
            if self.window_size % period != 0:
                # Pad sequence to make it divisible by period
                length = (self.window_size // period + 1) * period
                padding = torch.zeros([x.shape[0], length - self.window_size, x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.window_size
                out = x

            # Reshape time series into 2D representation [B, N, length // period, period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # Apply 2D convolution through Inception blocks
            out = self.conv(out)
            # Reshape back to original temporal dimension [B, length // period, period, N] -> [B, length, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # Remove padding if added
            res.append(out[:, :self.window_size, :])

        # Stack results from all periods
        res = torch.stack(res, dim=-1)  # Shape: [B, T, N, k]

        # Compute adaptive weights for period aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        # Weighted sum of multi-period features
        res = torch.sum(res * period_weight, -1)  # Shape: [B, T, N]
        # Add residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Complete TimesNet model for time series processing.
    
    Architecture:
    1. Data normalization
    2. Embedding layer
    3. Multiple TimesBlocks
    4. Layer normalization
    5. Projection layer
    6. Data denormalization
    
    Features:
    - Non-stationary data handling
    - Multi-scale temporal pattern modeling
    - Adaptive period learning
    """

    def __init__(self, configs: ConfigType):
        """
        Initialize TimesNet model.
        
        Args:
            configs: Configuration containing:
                window_size: Sequence length
                enc_in: Input dimension
                d_model: Model dimension
                c_out: Output dimension
                e_layers: Number of TimesBlocks
                embed_type: Type of embedding
                freq: Time frequency
                dropout: Dropout rate
        """
        super(Model, self).__init__()
        self.configs = configs
        self.window_size = configs.window_size
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.encoder_layers)])
        self.encoder_embedding = DataEmbedding(configs.input_channels, configs.d_model, configs.embedding_type,
                                               configs.freq,
                                           configs.dropout)
        self.layer = configs.encoder_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.output_channels, bias=True)

    def forward(self, x):
        """
        Forward pass of TimesNet.
        
        Process:
        1. Normalize input (mean-std normalization)
        2. Apply embedding
        3. Process through TimesBlocks
        4. Project to output dimension
        5. Denormalize output
        
        Args:
            x: Input time series [B, T, N]
        Returns:
            Processed time series [B, T, N]
        """
        # Normalization from Non-stationary Transformer
        # detach() prevents gradients from flowing back through the mean calculation
        # This is done because we only want to normalize the data, not learn the mean value
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        # When unbiased=False, use N as denominator to compute population variance
        # When unbiased=True, use (N-1) as denominator to compute sample variance
        std_deviation = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= std_deviation

        # embedding
        enc_out = self.encoder_embedding(x, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * std_deviation[:, 0, :].unsqueeze(1).repeat(1, self.window_size, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.window_size, 1)
        return dec_out
