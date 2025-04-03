"""
=================================================
@Author: Zenon
@Date: 2025-03-30
@Description：Multivariate Time Series Anomaly Detection by Capturing Coarse-Grained Intra- and Inter-Variate Dependencies
Paper link: https://arxiv.org/abs/2501.16364
Code Repository: https://github.com/ilwoof/MtsCID/
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

    def __init__(self, input_dimensions, d_model, win_size, patch_list, dropout=0.1, use_position_embedding=False):
        super().__init__()
        self.use_position_embedding = use_position_embedding

        component_network = ['real_part', 'imaginary_part']
        num_in_networks = len(component_network)

        self.fcl_layer = nn.ModuleList(
            [nn.Linear(input_dimensions, d_model, bias=False) for _ in range(num_in_networks)])
        self.fcl_normalization_layer = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_in_networks)])

        self.intra_variate_transformer_layer = nn.ModuleList(
            [ComplexAttentionLayer(d_model, 1, dropout=0.1) for _ in range(num_in_networks)])
        self.intra_variate_transformer_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_in_networks)])

        self.multiscale_temporal_attention = InceptionAttentionLayer(win_size, patch_list)

        if use_position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = torch.fft.rfft(x, dim=-2)
        x = complex_operator(self.fcl_layer, x)
        x = torch.fft.irfft(x, dim=-2)
        x = complex_operator(self.fcl_normalization_layer, x)
        residual = x

        x = torch.fft.rfft(x, dim=-2)
        x = complex_operator(self.intra_variate_transformer_layer, x)
        x = torch.fft.irfft(x, dim=-2)
        x = complex_operator(self.intra_variate_transformer_normalization_layer, x)
        x += residual

        x = complex_operator(self.multiscale_temporal_attention, x)

        if self.use_position_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class InterVariateEncoder(nn.Module):

    def __init__(self, input_dimensions, d_model, win_size, dropout, kernel_list, use_position_embedding=False):
        super().__init__()
        self.use_position_embedding = use_position_embedding

        component_network = ['real_part', 'imaginary_part']
        num_in_networks = len(component_network)

        self.multiscale_conv1d = InceptionBlock1d(input_dimensions, d_model, kernel_list, groups=1)
        self.multiscale_conv1d_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(win_size) for _ in range(num_in_networks)])

        w_model = win_size // 2 + 1
        self.inter_variate_transformer_layer = nn.ModuleList(
            [ComplexAttentionLayer(w_model, 1, dropout=0.1) for _ in range(num_in_networks)])
        self.inter_variate_transformer_normalization_layer = nn.ModuleList(
            [nn.LayerNorm(win_size) for _ in range(num_in_networks)])

        if use_position_embedding:
            self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = complex_operator(self.multiscale_conv1d, x)
        x = complex_operator(self.multiscale_conv1d_normalization_layer, x)

        x = torch.fft.rfft(x, dim=-1)
        x = complex_operator(self.inter_variate_transformer_layer, x)
        x = torch.fft.irfft(x, dim=-1)
        x = complex_operator(self.inter_variate_transformer_normalization_layer, x)

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

        return self.dropout(x)


class MtsCID(nn.Module):

    def __init__(self, configs: ConfigType, init_type='normal', gain=0.02):
        super().__init__()
        self.temperature = configs.temperature
        self.init_type = init_type
        self.gain = gain

        temporal_branch_dimension = configs.input_channels if self.temporal_brach_dimension_match == 'none' else configs.d_model
        inter_variate_branch_dimension = configs.input_channels if self.inter_variate_brach_dimension_match == 'none' else configs.d_model

        self.temporal_encoder = TemporalEncoder(configs.input_channels, temporal_branch_dimension,
                                                win_size=configs.window_size,
                                                patch_list=configs.patch_list, dropout=configs.dropout)
        self.inter_variate_encoder = InterVariateEncoder(configs.input_channels, inter_variate_branch_dimension,
                                                         win_size=configs.window_size,
                                                         dropout=configs.dropout, kernel_list=configs.kernel_list)

        self.activate_function = nn.GELU()
        self.memory_real, self.memory_imaginary = create_memory_matrix(inter_variate_branch_dimension,
                                                                       configs.window_size, 'sinusoid', 'options2')

        temporal_branch_output_dimension = configs.output_channels if self.temporal_brach_dimension_match == 'none' else configs.d_model

        self.weak_decoder = Decoder(temporal_branch_output_dimension, configs.output_channels, configs.window_size)

        if self.temporal_brach_dimension_match == 'none':
            self.feature_projector = nn.Identity()
        else:
            self.feature_projector = nn.Linear(temporal_branch_output_dimension, configs.output_channels)

        self.init_modules()

    def init_weights(self, m: nn.Module):
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

        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    def init_modules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_weights(m)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                self.init_weights(m)

    def forward(self, x):

        z1 = z2 = x
        device = x.device

        temporal_queries, _ = self.temporal_encoder(z1)
        inter_queries, _ = self.inter_variate_encoder(z2)
        memory = self.memory_real.T.to(device)

        attention = torch.einsum('blf,jl->bfj', inter_queries, self.memory_real.to(device).detach())
        attention = torch.softmax(attention / self.temperature, dim=-1)

        combined_z = self.feature_prj(temporal_queries)

        output, _ = self.weak_decoder(combined_z)

        return {"output": output, "queries": inter_queries, "memory": memory, "attention": attention}
