"""
=================================================
@Author: Zenon
@Date: 2025-03-27
@Description: Memory Layer Implementation
    This module implements a memory-augmented neural network layer that can store
    and retrieve information from an external memory matrix. The memory layer
    enhances model capabilities by:
    
    1. Storing representative patterns in memory slots
    2. Retrieving relevant information through attention mechanisms
    3. Updating memory contents during training
    
    This approach is particularly useful for anomaly detection in time series data
    as it can learn normal patterns and identify deviations from them.
==================================================
"""
import torch
import torch.nn.functional as F
from torch import nn

from tsadlib.utils.logger import logger


def hard_shrink_relu(attention, hyper_lambda=0.0025, epsilon=1e-12):
    """
    Apply hard shrinkage with ReLU activation to attention scores.
    
    This function implements a sparse attention mechanism by applying a threshold
    to attention scores, helping to focus on the most relevant memory items.
    
    Args:
        attention: Attention scores tensor
        hyper_lambda: Shrinkage threshold parameter
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Processed attention scores with enhanced sparsity
    """
    output = (F.relu(attention - hyper_lambda) * attention) / (torch.abs(attention - hyper_lambda) + epsilon)
    return output


class MemoryLayer(nn.Module):
    """
    Memory-augmented neural network layer.
    
    This layer maintains an external memory matrix that stores representative patterns.
    It can read from memory using attention mechanisms and update memory contents
    during training. The memory component helps capture normal patterns for anomaly detection.
    """

    def __init__(self, num_memory, feature_dimension, shrink_threshold=2.5e-3, memory_init_embedding=None,
                 mode='train'):
        """
        Initialize the memory layer.
        
        Args:
            num_memory: Number of memory slots (M)
            feature_dimension: Dimension of each memory slot (C)
            shrink_threshold: Threshold for hard shrinkage function
            memory_init_embedding: Pre-initialized memory embeddings
            mode: Operating mode ('train' or 'test')
        """
        super().__init__()

        self.num_memory = num_memory
        self.feature_dimension = feature_dimension  # C(=d_model)
        self.shrink_threshold = shrink_threshold
        self.mode = mode

        # Projection matrices for memory update mechanism
        self.U_projection = nn.Linear(feature_dimension, feature_dimension)
        self.W_projection = nn.Linear(feature_dimension, feature_dimension)

        if memory_init_embedding is None:
            # Initialize memory with random values for first training phase
            logger.info('loading memory item with random initialization (for first train phase)')
            # Normalize memory embeddings along feature dimension
            self.memory = F.normalize(torch.rand((self.num_memory, self.feature_dimension), dtype=torch.float), dim=1)
        else:
            # Use pre-trained memory for continued training or testing
            logger.info('loading memory item with first train\'s result (for second train or test phase)')
            self.memory = memory_init_embedding

    def get_attention_score(self, query, key):
        """
        Calculate attention scores between query and key.
        
        Args:
            query: Query tensor for attention calculation
            key: Key tensor for attention calculation
            
        Returns:
            Normalized attention scores after optional shrinkage
        """
        # Calculate raw attention scores through dot product
        attention = torch.matmul(query, torch.t(key))  # (TxC) x (CxM) -> TxM
        # Apply softmax to get attention distribution
        attention = F.softmax(attention, dim=-1)

        # Apply hard shrinkage if threshold is positive
        if self.shrink_threshold > 0:
            attention = hard_shrink_relu(attention, self.shrink_threshold)
            # Re-normalize attention scores after shrinkage
            attention = F.normalize(attention, p=1, dim=1)

        return attention

    def update(self, query):
        """
        Update memory contents based on input query.
        
        This method implements a gated update mechanism that selectively
        incorporates new information into memory slots.
        
        Args:
            query: Input features used to update memory
        """
        key = query.detach()
        # Calculate attention between memory and input
        attention = self.get_attention_score(self.memory, key)  # M x T

        # Weighted sum of input features based on attention
        add_memory = torch.matmul(attention, key)  # M x C

        # Gated update mechanism
        # Update gate determines how much to update each memory slot
        update_gate = torch.sigmoid(self.U_projection(self.memory) + self.W_projection(add_memory))  # M x C
        # Update memory with gated combination of old and new content
        self.memory = (1 - update_gate) * self.memory + update_gate * add_memory

    def read(self, query):
        """
        Read from memory based on query features.
        
        Args:
            query: Input features used to query memory
            
        Returns:
            Dictionary containing augmented features and attention scores
        """
        key = self.memory.detach()
        # Calculate attention between query and memory
        attention = self.get_attention_score(query, key)  # T x M
        # Retrieve memory content based on attention
        add_memory = torch.matmul(attention, key)  # T x C

        # Concatenate original query with retrieved memory content
        read_query = torch.cat((query, add_memory), dim=1)  # T x 2C

        return {'output': read_query, 'attention': attention}

    def forward(self, query):
        """
        Forward pass of the memory layer.
        
        Args:
            query: Input features to process with memory augmentation
            
        Returns:
            Dictionary containing:
            - output: Memory-augmented features
            - attention: Attention scores between input and memory
            - memory_init_embedding: Current memory state
        """
        # Get input shape information
        shape = query.data.shape
        dimensions = len(shape)

        # Reshape input for processing
        query = query.contiguous().view(-1, shape[-1])  # N x L x C or N x C -> T x C

        # Ensure memory is on the same device as input
        if query.device != self.memory.device:
            self.memory = self.memory.to(query.device)

        # Update memory during training
        if self.mode != 'test':
            self.update(query)

        # Read from memory to get augmented features
        outs = self.read(query)

        read_query, attention = outs['output'], outs['attention']

        # Reshape output based on input dimensions
        if dimensions == 2:
            pass  # No reshaping needed for 2D input
        elif dimensions == 3:
            # Reshape for 3D input (batch, sequence, features)
            read_query = read_query.view(shape[0], shape[1], 2 * shape[2])
            attention = attention.view(shape[0], shape[1], self.num_memory)
        else:
            raise TypeError('Wrong input dimension')

        return {'output': read_query, 'attention': attention, 'memory_init_embedding': self.memory}
