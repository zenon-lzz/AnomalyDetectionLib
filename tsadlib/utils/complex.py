"""
=================================================
@Author: Zenon
@Date: 2025-04-01
@Description: PyTorch Complex Number Operations Utilities
    This module provides extended support for complex-valued tensor operations in PyTorch,
    including neural network layer operations and mathematical functions that properly
    handle complex numbers.
==================================================
"""
import torch
from torch import nn


def complex_operator(net_layer, x):
    """Apply neural network layer operation to complex-valued input.
    
    Handles both real and complex inputs, applying the operation separately to
    real and imaginary components when needed. Supports ModuleList for separate
    real/imaginary processing.
    
    Args:
        net_layer: PyTorch layer or ModuleList of layers
        x: Input tensor (real or complex)
        
    Returns:
        Processed tensor with same dtype as input
    """
    if not torch.is_complex(x):
        return net_layer[0](x) if isinstance(net_layer, nn.ModuleList) else net_layer(x)
    else:
        # Special handling for LSTM due to tuple output
        if isinstance(net_layer[0], nn.LSTM):
            return torch.complex(net_layer[0](x.real)[0], net_layer[1](x.imag)[0]) if isinstance(net_layer,
                                                                                                 nn.ModuleList) else torch.complex(
                net_layer(x.real)[0], net_layer(x.imag)[0])
        else:
            return torch.complex(net_layer[0](x.real), net_layer[1](x.imag)) if isinstance(net_layer,
                                                                                           nn.ModuleList) else torch.complex(
                net_layer(x.real), net_layer(x.imag))


def complex_einsum(order, x, y):
    """Complex-aware einsum operation.
    
    Performs Einstein summation convention operations while properly handling
    complex-valued tensors according to complex arithmetic rules.
    
    Args:
        order: Einsum equation string
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Result tensor with proper complex multiplication if inputs are complex
    """
    x_flag = True
    y_flag = True
    if not torch.is_complex(x):
        x_flag = False
        x = torch.complex(x, torch.zeros_like(x).to(x.device))
    if not torch.is_complex(y):
        y_flag = False
        y = torch.complex(y, torch.zeros_like(y).to(y.device))
    if x_flag or y_flag:
        return torch.complex(torch.einsum(order, x.real, y.real) - torch.einsum(order, x.imag, y.imag),
                             torch.einsum(order, x.real, y.imag) + torch.einsum(order, x.imag, y.real))
    else:
        return torch.einsum(order, x.real, y.real)


def complex_softmax(x, dim=-1):
    """Complex-aware softmax operation.
    
    Applies softmax separately to real and imaginary components when input is complex.
    
    Args:
        x: Input tensor
        dim: Dimension along which softmax will be computed
        
    Returns:
        Softmax result preserving input dtype
    """
    if not torch.is_complex(x):
        return torch.softmax(x, dim=dim)
    else:
        return torch.complex(torch.softmax(x.real, dim=dim), torch.softmax(x.imag, dim=dim))


def complex_dropout(dropout_func, x):
    """Complex-aware dropout operation.
    
    Applies dropout mask consistently across real and imaginary components
    when input is complex-valued.
    
    Args:
        dropout_func: Initialized dropout layer
        x: Input tensor
        
    Returns:
        Dropout-applied tensor with same dtype as input
    """
    if not torch.is_complex(x):
        return dropout_func(x)
    else:
        return torch.complex(dropout_func(x.real), dropout_func(x.imag))


def complex_layer_normalization(norm_func, x):
    """Complex-aware layer normalization.
    
    Applies layer normalization separately to real and imaginary components
    when input is complex-valued.
    
    Args:
        norm_func: Initialized normalization layer
        x: Input tensor
        
    Returns:
        Normalized tensor with same dtype as input
    """
    if not torch.is_complex(x):
        return norm_func(x)
    else:
        return torch.complex(norm_func(x.real), norm_func(x.imag))
