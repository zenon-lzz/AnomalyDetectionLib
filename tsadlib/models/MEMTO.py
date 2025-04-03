"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection
Paper link: https://arxiv.org/abs/2312.02530
Code Repository: https://github.com/gunny97/MEMTO
==================================================
"""
import torch.nn.functional as F
from torch import nn

from tsadlib.configs.type import ConfigType
from tsadlib.layers.embeddings import DataEmbedding
from tsadlib.layers.memory_layer import MemoryLayer


class EncoderLayer(nn.Module):

    def __init__(self, attention_layer, d_model, dimension_fcl=None, dropout=0.1, activation='relu'):
        super().__init__()
        if dimension_fcl is None:
            dimension_fcl = 4 * d_model

        self.attention_layer = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dimension_fcl, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dimension_fcl, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        """

        :param x: [B x L x C(d_model)]
        :return:
        """
        out, _ = self.attention_layer(x, x, x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)  # N x L x C(=d_model)


class Encoder(nn.Module):
    def __init__(self, attention_layers, normalization=None):
        super().__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.normalization = normalization

    def forward(self, x):
        """

        :param x: [B x L x C(d_model)]
        :return:
        """
        for attention_layer in self.attention_layers:
            x = attention_layer(x)

        if self.normalization is not None:
            x = self.normalization(x)

        return x


class MEMTO(nn.Module):

    def __init__(self, configs: ConfigType, memory_init_embedding=None, shrink_threshold=0.):
        super().__init__()

        self.memory_init_embedding = memory_init_embedding
        self.embedding = DataEmbedding(configs.input_channels, configs.d_model, dropout=configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    nn.MultiheadAttention(embed_dim=configs.d_model, num_heads=configs.n_heads, dropout=configs.dropout,
                                          batch_first=True),
                    configs.d_model, configs.dimension_fcl, dropout=configs.dropout, activation='gelu'
                ) for _ in range(configs.encoder_layers)
            ],
            normalization=nn.LayerNorm(configs.d_model)
        )

        self.memory_layer = MemoryLayer(configs.num_memory, configs.d_model, shrink_threshold, memory_init_embedding,
                                        configs.mode)

        self.weak_decoder = nn.Linear(2 * configs.d_model, configs.output_channels)

    def forward(self, x):
        x = self.embedding(x)  # embedding : N x L x C(=d_model)
        queries = output = self.encoder(x)  # encoder out : N x L x C(=d_model)

        output_dict = self.memory_layer(output)
        output, attention, memory_item_embedding = output_dict['output'], output_dict['attention'], output_dict[
            'memory_init_embedding']

        memory = self.memory_layer.memory

        output = self.weak_decoder(output)
        return {"output": output, "queries": queries,
                "memory": memory,
                "attention": attention}
