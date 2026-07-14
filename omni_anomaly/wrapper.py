# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftplusStd(nn.Module):
    """Dense layer followed by softplus + epsilon for positive std."""

    def __init__(self, in_features, out_features, epsilon=1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.epsilon = epsilon

    def forward(self, x):
        return F.softplus(self.linear(x)) + self.epsilon


class StackedRNNEncoder(nn.Module):
    """
    GRU/LSTM encoder with post-RNN dense layers.
    Input: (batch, window_length, input_dim)
    Output: (batch, window_length, dense_dim)
    """

    def __init__(
        self,
        input_dim,
        rnn_num_hidden=500,
        rnn_cell='GRU',
        hidden_dense=2,
        dense_dim=500,
    ):
        super().__init__()
        self.window_length = None
        if rnn_cell == 'LSTM':
            self.rnn = nn.LSTM(input_dim, rnn_num_hidden, batch_first=True)
        elif rnn_cell == 'GRU':
            self.rnn = nn.GRU(input_dim, rnn_num_hidden, batch_first=True)
        elif rnn_cell == 'Basic':
            self.rnn = nn.RNN(input_dim, rnn_num_hidden, batch_first=True)
        else:
            raise ValueError('rnn_cell must be LSTM, GRU, or Basic')

        layers = []
        in_dim = rnn_num_hidden
        for _ in range(hidden_dense):
            layers.append(nn.Linear(in_dim, dense_dim))
            in_dim = dense_dim
        self.dense_layers = nn.ModuleList(layers)

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=0)
        outputs, _ = self.rnn(x)
        for layer in self.dense_layers:
            outputs = layer(outputs)
        return outputs


class GaussianParamNet(nn.Module):
    """RNN hidden states -> Gaussian mean and std."""

    def __init__(self, input_dim, output_dim, rnn_num_hidden, dense_dim,
                 rnn_cell='GRU', hidden_dense=2, std_epsilon=1e-4):
        super().__init__()
        self.encoder = StackedRNNEncoder(
            input_dim, rnn_num_hidden, rnn_cell, hidden_dense, dense_dim,
        )
        self.mean_layer = nn.Linear(dense_dim, output_dim)
        self.std_layer = SoftplusStd(dense_dim, output_dim, epsilon=std_epsilon)

    def forward(self, x):
        h = self.encoder(x)
        return {
            'mean': self.mean_layer(h),
            'std': self.std_layer(h),
        }


class RecurrentGaussianParamNet(nn.Module):
    """RNN hidden states -> input_q for RecurrentDistribution."""

    def __init__(self, input_dim, rnn_num_hidden, dense_dim,
                 rnn_cell='GRU', hidden_dense=2):
        super().__init__()
        self.encoder = StackedRNNEncoder(
            input_dim, rnn_num_hidden, rnn_cell, hidden_dense, dense_dim,
        )

    def forward(self, x):
        return {'input_q': self.encoder(x)}
