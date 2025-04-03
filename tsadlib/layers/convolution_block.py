"""
=================================================
@Author: Zenon
@Date: 2025-03-15
@Description: Multi-scale convolutional blocks for time series processing

This module implements various Inception-style convolutional blocks adapted for:
- 2D feature extraction (InceptionBlockV1/V2)
- 1D time series processing (InceptionBlock1d)

Key Features:
1. InceptionBlockV1: Symmetric square kernels for uniform feature extraction
2. InceptionBlockV2: Asymmetric rectangular kernels for directional patterns  
3. InceptionBlock1d: Specialized 1D convolutions for temporal data with circular padding
==================================================
"""
import torch
from torch import nn


class InceptionBlockV1(nn.Module):
    """Inception block with symmetric convolutions.
    
    This version uses square kernels (same height and width) for convolution operations.
    Each kernel processes the input symmetrically in both spatial dimensions.

    Based on GoogLeNet/Inception-v1 design philosophy from the paper 'Going Deeper with Convolutions'
    Core idea: Extract multi-scale features through parallel use of different sized convolution kernels
    Suitable for handling in time series anomaly detection:
    - Features with strong temporal local correlation
    - Patterns requiring consideration of multiple time scales
    - Direction-invariant temporal patterns
    
    Key characteristics:
    - Uses square kernels: kernel_size = 2i + 1 (where i is kernel index)
    - Same padding in both dimensions
    - Processes features uniformly across both spatial dimensions
    """

    def __init__(self, input_channels, output_channels, num_kernels=6, init_weight=True):
        """Initialize the Inception block V1.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            num_kernels: Number of different kernel sizes to use
            init_weight: Whether to initialize weights using Kaiming initialization
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_kernels = num_kernels
        kernels = []
        # Create kernels with increasing size: 1x1, 3x3, 5x5, 7x7, 9x9, 11x11
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(input_channels, output_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the inception block.
        
        Applies each kernel to input and averages the results.
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionBlockV2(nn.Module):
    """Inception block with asymmetric convolutions.
    
    This version uses rectangular kernels (different height and width) for convolution operations.
    Kernels are split into vertical and horizontal components to capture directional patterns.

    Based on Inception-v2/v3's asymmetric convolution factorization concept from the paper 'Rethinking the Inception Architecture for Computer Vision'
    Core idea: Factorize large convolution kernels to reduce computational complexity while maintaining receptive field
    Suitable for handling in time series anomaly detection:
    - Temporal patterns with clear directionality
    - Scenarios with limited computational resources
    - Cases requiring separate attention to short-term and long-term dependencies
    
    Key characteristics:
    - Uses rectangular kernels: [1×(2i+3)] and [(2i+3)×1]
    - Different padding for height and width
    - Processes features separately in vertical and horizontal directions
    - Includes an additional 1x1 convolution
    """

    def __init__(self, input_channels, output_channels, num_kernels=6, init_weight=True):
        """Initialize the Inception block V2.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            num_kernels: Total number of kernel pairs (vertical + horizontal)
            init_weight: Whether to initialize weights using Kaiming initialization
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_kernels = num_kernels
        kernels = []
        # Create pairs of kernels: one vertical and one horizontal
        for i in range(self.num_kernels // 2):
            # Horizontal kernel: 1×(2i+3)
            kernels.append(nn.Conv2d(input_channels, output_channels, kernel_size=(1, 2 * i + 3), padding=(0, i + 1)))
            # Vertical kernel: (2i+3)×1
            kernels.append(nn.Conv2d(input_channels, output_channels, kernel_size=(2 * i + 3, 1), padding=(i + 1, 0)))
        # Additional 1x1 convolution for point-wise processing
        kernels.append(nn.Conv2d(input_channels, output_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the inception block.
        
        Applies all kernels (horizontal, vertical, and 1x1) to input and averages the results.
        """
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):  # +1 for the 1x1 convolution
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionBlock1d(nn.Module):
    """1D Inception-style convolutional block for temporal pattern extraction.
    
    Processes 1D time series data using parallel convolutions with different kernel sizes
    to capture patterns at multiple temporal scales. Uses circular padding to maintain
    temporal continuity at sequence boundaries.
    
    Typical Use Cases:
    - Multi-scale time series feature extraction
    - Anomaly detection in temporal sequences
    - Processing of periodic/cyclical time series data

    Architecture Highlights:
    - Multiple parallel 1D convolutions with configurable kernel sizes
    - Circular padding for boundary continuity
    - Grouped convolutions for parameter efficiency
    - Kaiming weight initialization
    """

    def __init__(self, in_channels, out_channels, kernel_list, groups=1, init_weight=True):
        """Initialize the 1D Inception block.
        
        Args:
            in_channels (int): Number of input channels/features
            out_channels (int): Number of output channels/features
            kernel_list (list[int]): List of kernel sizes for parallel convolutions
            groups (int): Number of groups for grouped convolution (default: 1)
            init_weight (bool): Whether to initialize weights (default: True)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_list = kernel_list

        # Initialize parallel convolution branches
        kernels = []
        for kernel_size in self.kernel_list:
            kernels.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding='same',  # Maintains input sequence length
                padding_mode='circular',  # Preserves temporal continuity
                bias=False,  # Omit bias for efficiency
                groups=groups  # Enable grouped convolutions
            ))
        self.convolutions = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization scheme.
        
        Uses ReLU nonlinear initialization for all 1D convolutional layers.
        Initializes biases to zero if present (though typically disabled).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the 1D Inception block.
        
        Args:
            x (Tensor): Input tensor of shape [batch, channels, time_steps]
            
        Returns:
            Tensor: Output features combining multi-scale patterns
                   Shape: [batch, out_channels, time_steps]
                   
        Processing Steps:
        1. Apply each parallel convolution branch
        2. Combine results through averaging
        3. Maintain original temporal resolution
        """
        # Process through all convolution branches
        res_list = []
        for conv in self.convolutions:
            res_list.append(conv(x))

        # Average multi-scale features
        return torch.stack(res_list, dim=-1).mean(-1)
