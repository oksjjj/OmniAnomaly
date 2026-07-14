# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """Planar normalizing flow: z' = z + u * tanh(w^T z + b)."""

    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        """
        Args:
            z: (..., dim)
        Returns:
            z_out, log_det (scalar per batch element summed over extra dims)
        """
        wz_b = (z * self.w).sum(dim=-1, keepdim=True) + self.b
        tanh_wz_b = torch.tanh(wz_b)
        z_out = z + self.u * tanh_wz_b

        psi = (1.0 - tanh_wz_b ** 2) * self.w
        det = 1.0 + (psi * self.u).sum(dim=-1)
        log_det = torch.log(torch.abs(det) + 1e-8)
        return z_out, log_det


class PlanarNormalizingFlows(nn.Module):
    """Stack of planar flows (matches tfsnippet planar_normalizing_flows)."""

    def __init__(self, dim, n_layers):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_layers)])

    def forward(self, z):
        log_det = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, ld = flow(z)
            log_det = log_det + ld
        return z, log_det
