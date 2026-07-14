# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn

from omni_anomaly.distributions import gaussian_log_prob


LOG_2PI = math.log(2 * math.pi)


class RecurrentDistribution(nn.Module):
    """
    Recurrent variational distribution q(z|x).
    Matches the TensorFlow RecurrentDistribution implementation.
    """

    def __init__(self, z_dim, window_length, input_q_dim, std_epsilon=1e-4):
        super().__init__()
        self.z_dim = z_dim
        self.window_length = window_length
        self.input_q_dim = input_q_dim
        concat_dim = input_q_dim + z_dim
        self.mean_layer = nn.Linear(concat_dim, z_dim)
        self.std_layer = nn.Sequential(
            nn.Linear(concat_dim, z_dim),
            nn.Softplus(),
        )
        self.std_epsilon = std_epsilon

    def _std(self, inp):
        return self.std_layer(inp) + self.std_epsilon

    def sample(self, input_q, n_samples=None):
        """
        Args:
            input_q: (batch, window_length, input_q_dim)
            n_samples: number of MC samples (None -> 1)
        Returns:
            z: (batch, window_length, z_dim) or (n_samples, batch, window_length, z_dim)
        """
        batch_size = input_q.shape[0]
        device = input_q.device
        dtype = input_q.dtype

        squeeze = n_samples is None
        if n_samples is None:
            n_samples = 1

        z_steps = []
        z_prev = torch.zeros(n_samples, batch_size, self.z_dim, device=device, dtype=dtype)

        for t in range(self.window_length):
            input_q_t = input_q[:, t, :].unsqueeze(0).expand(n_samples, -1, -1)
            inp = torch.cat([input_q_t, z_prev], dim=-1)
            mu = self.mean_layer(inp)
            std = self._std(inp)
            noise = torch.randn_like(mu)
            z_t = mu + noise * std
            z_steps.append(z_t)
            z_prev = z_t

        samples = torch.stack(z_steps, dim=2)  # n_samples, batch, time, z_dim

        if squeeze:
            return samples.squeeze(0)
        return samples

    def log_prob(self, z, input_q, group_ndims=1):
        """
        Args:
            z: (batch, window_length, z_dim) or (n_samples, batch, window_length, z_dim)
            input_q: (batch, window_length, input_q_dim)
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        n_samples, batch_size, window_length, _ = z.shape

        log_probs = []
        for t in range(window_length):
            z_t = z[:, :, t, :]
            input_q_t = input_q[:, t, :].unsqueeze(0).expand(n_samples, -1, -1)
            inp = torch.cat([z_t, input_q_t], dim=-1)
            mu = self.mean_layer(inp)
            std = self._std(inp)
            lp = gaussian_log_prob(z_t, mu, std, group_ndims=1)
            log_probs.append(lp)

        log_prob = torch.stack(log_probs, dim=-1)
        if n_samples == 1:
            log_prob = log_prob.squeeze(0)
        if group_ndims == 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob
