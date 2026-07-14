# -*- coding: utf-8 -*-
import math

import torch


LOG_2PI = math.log(2 * math.pi)


def gaussian_log_prob(x, mean, std, group_ndims=1):
    """
    Diagonal Gaussian log probability.

    Args:
        x, mean, std: broadcast-compatible tensors.
        group_ndims: 0 = no sum; 1 = sum over last dim; 2 = sum over last two dims.
    """
    precision = 1.0 / (std ** 2 + 1e-8)
    log_prob = (
        -0.5 * LOG_2PI
        - torch.log(std + 1e-8)
        - 0.5 * precision * torch.clamp((x - mean) ** 2, max=1e16)
    )
    if group_ndims == 1:
        log_prob = log_prob.sum(dim=-1)
    elif group_ndims == 2:
        log_prob = log_prob.sum(dim=-1).sum(dim=-1)
    return log_prob


class DiagonalNormal:
    """Diagonal Normal distribution helper."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def log_prob(self, x, group_ndims=1):
        return gaussian_log_prob(x, self.mean, self.std, group_ndims=group_ndims)

    def sample(self, n_samples=None):
        if n_samples is None or n_samples < 2:
            noise = torch.randn_like(self.mean)
            return self.mean + self.std * noise
        noise = torch.randn(
            n_samples, *self.mean.shape,
            device=self.mean.device, dtype=self.mean.dtype,
        )
        return self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise


class GaussianStateSpacePrior:
    """
    Linear Gaussian state-space prior: z_t = z_{t-1} + eps, z_0 ~ N(0, I).
    Matches TFP LinearGaussianStateSpaceModel with identity transition.
    """

    def __init__(self, z_dim, window_length):
        self.z_dim = z_dim
        self.window_length = window_length

    def log_prob(self, z, group_ndims=1):
        """
        Args:
            z: (batch, window_length, z_dim) or (n_samples, batch, window_length, z_dim)
        """
        z0_log_prob = gaussian_log_prob(
            z[..., 0, :],
            torch.zeros_like(z[..., 0, :]),
            torch.ones_like(z[..., 0, :]),
            group_ndims=1,
        )
        if z.shape[-2] == 1:
            return z0_log_prob

        transition_log_prob = gaussian_log_prob(
            z[..., 1:, :],
            z[..., :-1, :],
            torch.ones_like(z[..., 1:, :]),
            group_ndims=1,
        )
        # transition_log_prob: (..., window_length-1)
        transition_log_prob = transition_log_prob.sum(dim=-1)
        return z0_log_prob + transition_log_prob

    def sample(self, batch_size, n_samples=None, device=None, dtype=torch.float32):
        shape = (batch_size, self.window_length, self.z_dim)
        z = torch.zeros(*shape, device=device, dtype=dtype)
        z[:, 0, :] = torch.randn(batch_size, self.z_dim, device=device, dtype=dtype)
        for t in range(1, self.window_length):
            z[:, t, :] = z[:, t - 1, :] + torch.randn(
                batch_size, self.z_dim, device=device, dtype=dtype,
            )
        if n_samples is not None and n_samples > 1:
            samples = [self.sample(batch_size, None, device, dtype) for _ in range(n_samples)]
            return torch.stack(samples, dim=0)
        return z
