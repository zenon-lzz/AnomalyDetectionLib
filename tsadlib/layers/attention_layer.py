"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description: Attention Mechanism Layer Implementation
==================================================
"""

import torch
from einops import rearrange
from torch import nn

from tsadlib.utils.complex import complex_einsum, complex_dropout, complex_softmax


class ComplexAttentionLayer(nn.Module):
    """Complex-valued multi-head attention layer.
    
    Implements attention mechanism that operates on complex-valued tensors,
    preserving phase information while computing attention weights.
    
    Args:
        d_model: Dimension of input features
        n_heads: Number of attention heads
        scale: Optional scaling factor for attention scores
        dropout: Dropout probability for attention weights
    """

    def __init__(self, d_model, n_heads, scale=None, dropout=0.0):
        """Initialize attention layer parameters."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Ensure number of heads divides feature dimension evenly
        self.n_heads = n_heads if (d_model % n_heads) == 0 else 1

        # Calculate head dimension and scaling factor
        z = d_model if n_heads == 1 else (d_model // n_heads)
        self.scale = scale if scale is not None else 1. / torch.sqrt(z)

    def forward(self, input_data):
        """Compute attention-weighted output.
        
        Args:
            input_data: Complex-valued input tensor [batch, length, features]
            
        Returns:
            tuple: (attention_output, attention_weights)
                   attention_output: [batch, length, features]
                   attention_weights: [batch, length, length] (averaged over heads)
        """
        batch, length, _ = input_data.shape

        # Split features into multiple heads
        queries = input_data.contiguous().view(batch, length, self.n_heads, -1)
        keys = input_data.contiguous().view(batch, length, self.n_heads, -1)
        values = input_data.contiguous().view(batch, length, self.n_heads, -1)

        # Compute complex-valued attention scores
        attention_scores = complex_einsum('nlhd,nshd->nhls', queries, keys)

        # Apply scaling and softmax
        attention_weights = complex_softmax(self.scale * attention_scores, dim=-1)

        # Compute weighted sum of values
        output = complex_einsum('nhls,nshd->nlhd', attention_weights, values).contiguous()

        # Average attention weights across heads
        attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=-2)

        return complex_dropout(self.dropout, output), attention_weights


class InceptionAttentionLayer(nn.Module):
    """Multi-scale attention layer with patch-based processing.
    
    Implements an inception-style architecture that applies attention mechanisms
    at multiple scales (patch sizes) and combines the results. Particularly useful
    for capturing patterns at different temporal resolutions.
    
    Args:
        win_size: Total window size of input sequences
        patch_list: List of patch sizes to process (must divide win_size evenly)
        init_weight: Whether to initialize weights with Kaiming normal (default: True)
    """

    def __init__(self, win_size, patch_list, init_weight=True):
        """Initialize patch attention layers and linear transformations."""
        super().__init__()
        self.patch_list = patch_list

        # Create attention layers and linear projections for each patch size
        patch_attention_layers = []
        linear_layers = []
        for patch_size in self.patch_list:
            patch_number = win_size // patch_size
            # Single-head attention for each patch
            patch_attention_layers.append(ComplexAttentionLayer(d_model=patch_size, n_heads=1))
            # Linear projection to original patch size
            linear_layers.append(nn.Linear(patch_number, patch_size))

        self.patch_attention_layers = nn.ModuleList(patch_attention_layers)
        self.linear_layers = nn.ModuleList(linear_layers)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal and zero biases.
        
        Applies to all linear layers in the module. Uses ReLU nonlinear
        initialization scheme.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Process input through multi-scale attention pipeline.
        
        Args:
            x: Input tensor of shape [batch, seq_len, features]
            
        Returns:
            Combined multi-scale attention output of shape [batch, seq_len, features]
        """
        batch, _, _ = x.size()
        res_list = []

        # Process each patch size separately
        for i, p_size in enumerate(self.patch_list):
            # Reshape into [batch*features, num_patches, patch_size]
            z = rearrange(x, 'b (w p) c -> (b c) w p', p=p_size).contiguous()

            # Apply attention and get attention-weighted output
            _, z = self.patch_attention_layers[i](z)

            # Project back to original patch dimension
            z = self.linear_layers[i](z)

            # Reshape back to original format
            z = rearrange(z, '(b c) w p -> b (w p) c', b=batch).contiguous()
            res_list.append(z)

        # Combine results from all scales
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
