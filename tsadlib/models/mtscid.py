"""
=================================================
@Author: Zenon
@Date: 2025-03-30
@Description：Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies
Paper link: https://arxiv.org/abs/2501.16364
Code Repository: https://github.com/ilwoof/MtsCID/

A dual-branch neural network architecture for detecting anomalies in multivariate
time series by capturing both intra-variable temporal patterns and inter-variable
dependencies using frequency-domain transformations and attention mechanisms.
==================================================
"""
import torch
from torch import nn

from tsadlib.configs.type import ConfigType
from tsadlib.layers.attention_layer import ComplexAttentionLayer, InceptionAttentionLayer
from tsadlib.layers.convolution_block import InceptionBlock1d
from tsadlib.layers.embeddings import PositionalEmbedding
from tsadlib.layers.memory_layer import create_memory_matrix
from tsadlib.utils.complex import complex_operator


class TemporalEncoder(nn.Module):
    """Encodes temporal patterns within individual time series variables.
    
    This module processes time series data through:
    1. Frequency-domain transformations (FFT)
    2. Complex-valued neural network operations
    3. Multi-scale temporal attention
    
    Args:
        input_dimensions (int): Number of input features/channels
        d_model (int): Dimension of model embeddings
        win_size (int): Size of input window
        patch_list (list): List of patch sizes for multi-scale processing
        dropout (float): Dropout probability (default: 0.1)
        use_position_embedding (bool): Whether to add positional embeddings (default: False)
    """

    def __init__(self, input_dimensions, d_model, win_size, patch_list, dropout=0.1, use_position_embedding=False):
        """Initialize the TemporalEncoder with specified parameters."""
        super().__init__()
        self.use_position_embedding = use_position_embedding

        # Define components for complex-valued processing (real and imaginary parts)
        component_network = ['real_part', 'imaginary_part']
        num_in_networks = len(component_network)

        # Initialize feed-forward layers for both real and imaginary components
        self.fcl_layer = nn.ModuleList(
            [nn.Linear(input_dimensions, d_model, bias=False) for _ in range(num_in_networks)])
        self.fcl_normalization_layer = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_in_networks)])

        # Initialize attention layers for intra-variate dependencies
        self.intra_variate_transformer_layer = nn.ModuleList(
            [ComplexAttentionLayer(d_model, 1, dropout=0.1, need_weights=False) for _ in range(num_in_networks)])
        self.intra_variate_transformer_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_in_networks)])

        # Multi-scale temporal attention mechanism
        self.multiscale_temporal_attention = InceptionAttentionLayer(win_size, patch_list)

        # Optional positional embeddings
        if use_position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Process input through temporal encoding pipeline.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, features]
            
        Returns:
            Tensor: Encoded temporal representations with same shape as input
        """
        # Transform to frequency domain
        x = torch.fft.rfft(x, dim=-2)
        
        # Process through feed-forward layers
        x = complex_operator(self.fcl_layer, x)
        x = torch.fft.irfft(x, dim=-2)
        x = complex_operator(self.fcl_normalization_layer, x)
        residual = x  # Save for residual connection

        # Intra-variate attention in frequency domain
        x = torch.fft.rfft(x, dim=-2)
        x = complex_operator(self.intra_variate_transformer_layer, x)
        x = torch.fft.irfft(x, dim=-2)
        x = complex_operator(self.intra_variate_transformer_normalization_layer, x)
        x += residual  # Add residual connection

        # Apply multi-scale temporal attention
        x = complex_operator(self.multiscale_temporal_attention, x)

        # Add positional embeddings if enabled
        if self.use_position_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class InterVariateEncoder(nn.Module):
    """Encodes relationships between different time series variables.
    
    This module processes cross-variable dependencies through:
    1. Multi-scale 1D convolutions
    2. Frequency-domain attention mechanisms
    3. Complex-valued neural network operations
    
    Args:
        input_dimensions (int): Number of input features/channels
        d_model (int): Dimension of model embeddings
        win_size (int): Size of input window
        dropout (float): Dropout probability
        kernel_list (list): List of kernel sizes for multi-scale convolution
        use_position_embedding (bool): Whether to add positional embeddings (default: False)
    """

    def __init__(self, input_dimensions, d_model, win_size, dropout, kernel_list, use_position_embedding=False):
        """Initialize the InterVariateEncoder with specified parameters."""
        super().__init__()
        self.use_position_embedding = use_position_embedding

        # Define components for complex-valued processing (real and imaginary parts)
        component_network = ['real_part', 'imaginary_part']
        num_in_networks = len(component_network)

        # Initialize multi-scale convolution layers
        self.multiscale_conv1d = InceptionBlock1d(input_dimensions, d_model, kernel_list, groups=1)
        self.multiscale_conv1d_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(win_size) for _ in range(num_in_networks)])

        # Calculate frequency dimension for attention
        w_model = win_size // 2 + 1
        
        # Initialize attention layers for inter-variate dependencies
        self.inter_variate_transformer_layer = nn.ModuleList(
            [ComplexAttentionLayer(w_model, 1, dropout=0.1, need_weights=False) for _ in range(num_in_networks)])
        self.inter_variate_transformer_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(win_size) for _ in range(num_in_networks)])

        # Optional positional embeddings
        if use_position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Process input through inter-variate encoding pipeline.
        
        Args:
            x (Tensor): Input tensor of shape [batch, seq_len, features]
            
        Returns:
            Tensor: Encoded inter-variate representations with same shape as input
        """
        # Permute dimensions for convolution operation
        x = x.permute(0, 2, 1)
        
        # Apply multi-scale convolutions
        x = complex_operator(self.multiscale_conv1d, x)
        x = complex_operator(self.multiscale_conv1d_normalization_layer, x)

        # Transform to frequency domain and apply attention
        x = torch.fft.rfft(x, dim=-1)
        x = complex_operator(self.inter_variate_transformer_layer, x)
        x = torch.fft.irfft(x, dim=-1)
        x = complex_operator(self.inter_variate_transformer_normalization_layer, x)
        
        # Restore original dimension order
        x = x.permute(0, 2, 1)

        # Add positional embeddings if enabled
        if self.use_position_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class Decoder(nn.Module):

    def __init__(self, d_model, output_channels):
        super().__init__()

        component_network = ['real_part', 'imaginary_part']
        num_in_networks = len(component_network)
        self.projector = nn.ModuleList(
            [nn.Linear(d_model, output_channels, bias=False) for _ in range(num_in_networks)])
        self.normalization_layer = nn.ModuleList([nn.LayerNorm(output_channels) for _ in range(num_in_networks)])

    def forward(self, x):
        x = complex_operator(self.projector, x)
        x = complex_operator(self.normalization_layer, x)

        return x


class MtsCID(nn.Module):
    """Main MtsCID model combining temporal and inter-variate encoders.
    
    Implements the complete anomaly detection pipeline:
    1. Temporal pattern encoding
    2. Cross-variable dependency modeling
    3. Memory-augmented attention
    4. Input reconstruction
    
    Args:
        configs (ConfigType): Model configuration parameters
        init_type (str): Weight initialization method ('normal', 'xavier', etc.) (default: 'normal')
        gain (float): Scaling factor for initialization (default: 0.02)
    """

    def __init__(self, configs: ConfigType, init_type='normal', gain=0.02):
        """Initialize the MtsCID model with configuration parameters."""
        super().__init__()
        self.temperature = configs.temperature  # Temperature parameter for attention
        self.init_type = init_type  # Weight initialization method
        self.gain = gain  # Initialization scaling factor

        # Initialize dimensions for both branches
        temporal_branch_dimension = configs.input_channels
        inter_variate_branch_dimension = configs.input_channels

        # Initialize temporal and inter-variate encoders
        self.temporal_encoder = TemporalEncoder(
            configs.input_channels, temporal_branch_dimension,
            win_size=configs.window_size,
            patch_list=configs.patch_list, dropout=configs.dropout)
        
        self.inter_variate_encoder = InterVariateEncoder(
            configs.input_channels, inter_variate_branch_dimension,
            win_size=configs.window_size,
            dropout=configs.dropout, kernel_list=configs.kernel_list)

        self.activate_function = nn.GELU()  # Activation function
        
        # Initialize memory matrix for attention mechanism
        self.memory_real, self.memory_imaginary = create_memory_matrix(
            inter_variate_branch_dimension,
            configs.window_size, 'sinusoid', 'options2')

        # Initialize decoder for reconstruction
        temporal_branch_output_dimension = configs.output_channels
        self.weak_decoder = Decoder(temporal_branch_output_dimension, configs.output_channels)

        # Optional feature projector (currently commented out)
        # if self.temporal_brach_dimension_match == 'none':
        #     self.feature_projector = nn.Identity()
        # else:
        #     self.feature_projector = nn.Linear(temporal_branch_output_dimension, configs.output_channels)

        self.init_modules()  # Initialize all model weights

    def init_weights(self, m: nn.Module):
        """Initialize weights for a given module.
        
        Supports multiple initialization schemes:
        - 'normal': Normal distribution initialization
        - 'xavier': Xavier/Glorot initialization
        - 'kaiming': Kaiming/He initialization
        - 'orthogonal': Orthogonal initialization
        - Default: Uniform initialization
        
        Args:
            m (nn.Module): Module to initialize
        """
        if self.init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, self.gain)
        elif self.init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=self.gain)
        elif self.init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif self.init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=self.gain)
        else:
            torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

        # Initialize biases to zero if they exist
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    def init_modules(self):
        """Initialize weights for all trainable modules in the network."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_weights(m)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                self.init_weights(m)

    def forward(self, x):
        """Complete forward pass through MtsCID model.
        
        Args:
            x (Tensor): Input time series of shape [batch, seq_len, features]
            
        Returns:
            dict: Dictionary containing:
                - output: Reconstructed time series
                - queries: Learned representations for anomaly scoring
                - memory: Memory matrix values
                - attention: Attention weights for interpretability
        """
        z1 = z2 = x  # Process same input through both branches
        device = x.device  # Get device for memory operations

        # Process through both encoders
        temporal_queries = self.temporal_encoder(z1)
        inter_queries = self.inter_variate_encoder(z2)
        
        # Prepare memory matrix for attention
        memory = self.memory_real.T.to(device)

        # Compute attention scores
        attention = torch.einsum('blf,jl->bfj', inter_queries, self.memory_real.to(device).detach())
        attention = torch.softmax(attention / self.temperature, dim=-1)

        # Combine features (currently using direct temporal queries)
        # combined_z = self.feature_prj(temporal_queries)
        combined_z = temporal_queries

        # Reconstruct input
        output = self.weak_decoder(combined_z)

        return {
            "output": output,
            "queries": inter_queries,
            "memory": memory,
            "attention": attention
        }
