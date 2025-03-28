"""
=================================================
@Author: Zenon
@Date: 2025-03-26
@Description：MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection
Paper link: https://arxiv.org/abs/2312.02530
Code Repository: https://github.com/gunny97/MEMTO.git
==================================================
"""
import torch.nn.functional as F
from torch import nn

from tsadlib.configs.type import ConfigType
from tsadlib.layers.attention_layer import AttentionLayer
from tsadlib.layers.embeddings import DataEmbedding
from tsadlib.layers.memory_layer import MemoryLayer


class EncoderLayer(nn.Module):

    def __init__(self, attention_layer, d_model, dimension_fcl=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
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
        out = self.attention_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)  # N x L x C(=d_model)


class Encoder(nn.Module):
    def __init__(self, attention_layers, normalization=None):
        super(Encoder, self).__init__()
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


class Model(nn.Module):

    def __init__(self, configs: ConfigType, shrink_threshold=0.):
        super(Model, self).__init__()

        self.memory_initial = configs.memory_initial
        self.embedding = DataEmbedding(configs.input_channels, configs.d_model, dropout=configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(configs.window_size, configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model, configs.dimension_fcl, dropout=configs.dropout, activation='gelu'
                ) for _ in range(configs.encoder_layers)
            ],
            normalization=nn.LayerNorm(configs.d_model)
        )

        self.memory_layer = MemoryLayer(configs.num_memory, configs.d_model, shrink_threshold, configs.memory_initial,
                                        configs.mode)

        self.weak_decoder = nn.Linear(configs.d_model, configs.output_channels)

    def forward(self, x):
        x = self.embedding(x)  # embedding : N x L x C(=d_model)
        queries = out = self.encoder(x)  # encoder out : N x L x C(=d_model)

        outputs = self.memory_layer(out)
        out, attention, memory_item_embedding = outputs['output'], outputs['attention'], outputs[
            'memory_init_embedding']

        memory = self.memory_layer.memory

        if self.memory_initial:
            return {"out": out, "memory_item_embedding": None, "queries": queries, "memory": memory}
        else:

            out = self.weak_decoder(out)
            return {"out": out, "memory_item_embedding": memory_item_embedding, "queries": queries, "memory": memory,
                    "attn": attention}
