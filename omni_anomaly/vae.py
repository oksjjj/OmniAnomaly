# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from omni_anomaly.distributions import DiagonalNormal, GaussianStateSpacePrior, gaussian_log_prob
from omni_anomaly.recurrent_distribution import RecurrentDistribution


class VAE(nn.Module):
    """
    Variational Auto-Encoder for OmniAnomaly.
    PyTorch port of the original tfsnippet-based VAE.
    """

    def __init__(
        self,
        config,
        h_for_p_x,
        h_for_q_z,
        use_connected_z_p=True,
        use_connected_z_q=True,
    ):
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        self.x_dim = config.x_dim
        self.window_length = config.window_length
        self.use_connected_z_p = use_connected_z_p
        self.use_connected_z_q = use_connected_z_q

        self.h_for_p_x = h_for_p_x
        self.h_for_q_z = h_for_q_z

        if use_connected_z_p:
            self.p_z = GaussianStateSpacePrior(config.z_dim, config.window_length)
        else:
            self.p_z = None

        if use_connected_z_q:
            self.recurrent_q = RecurrentDistribution(
                z_dim=config.z_dim,
                window_length=config.window_length,
                input_q_dim=config.dense_dim,
                std_epsilon=config.std_epsilon,
            )
        else:
            self.recurrent_q = None

    def _make_normal(self, params):
        return DiagonalNormal(params['mean'], params['std'])

    def variational(self, x, n_z=None, posterior_flow=None):
        """
        q(z|x) network.

        Returns dict with keys: z, log_q, input_q (optional)
        """
        z_params = self.h_for_q_z(x)

        if self.use_connected_z_q:
            input_q = z_params['input_q']
            z = self.recurrent_q.sample(input_q, n_samples=n_z)
            log_q = self.recurrent_q.log_prob(z, input_q, group_ndims=1)
            if n_z is not None and n_z > 1:
                log_q = log_q.sum(dim=-1)
        else:
            normal = self._make_normal(z_params)
            z = normal.sample(n_samples=n_z)
            log_q = normal.log_prob(z if n_z is None else z, group_ndims=1)
            if n_z is not None and n_z > 1:
                log_q = log_q.sum(dim=-1)

        log_det = 0.0
        if posterior_flow is not None:
            orig_shape = z.shape
            if z.dim() == 4:
                n_s, batch, time, dim = z.shape
                z_flat = z.reshape(n_s * batch * time, dim)
                z_flat, ld = posterior_flow(z_flat)
                z = z_flat.reshape(n_s, batch, time, dim)
                log_det = ld.reshape(n_s, batch, time).sum(dim=-1)
            else:
                batch, time, dim = z.shape
                z_flat = z.reshape(batch * time, dim)
                z_flat, ld = posterior_flow(z_flat)
                z = z_flat.reshape(batch, time, dim)
                log_det = ld.reshape(batch, time).sum(dim=-1)
            log_q = log_q - log_det

        return {'z': z, 'log_q': log_q, 'z_params': z_params}

    def model_net(self, z, x=None, n_z=None):
        """
        p(x|z) and p(z) network.

        Args:
            z: latent samples from q
            x: observed x (for computing reconstruction log prob)
        """
        if self.use_connected_z_p:
            log_p_z = self.p_z.log_prob(z, group_ndims=1)
            if n_z is not None and z.dim() == 4:
                log_p_z = log_p_z.sum(dim=-1) if log_p_z.dim() > 1 else log_p_z
        else:
            if z.dim() == 4:
                z_for_prior = z
            else:
                z_for_prior = z
            log_p_z = gaussian_log_prob(
                z_for_prior,
                torch.zeros_like(z_for_prior),
                torch.ones_like(z_for_prior),
                group_ndims=1,
            )
            if z.dim() == 4:
                log_p_z = log_p_z.sum(dim=(-1, -2))
            else:
                log_p_z = log_p_z.sum(dim=-1)

        x_params = self.h_for_p_x(z)
        p_x = self._make_normal(x_params)

        if x is not None:
            if z.dim() == 4:
                x_obs = x.unsqueeze(0)
                log_p_x = p_x.log_prob(x_obs, group_ndims=1)
                log_p_x = log_p_x.sum(dim=-1)
            else:
                log_p_x = p_x.log_prob(x, group_ndims=1)
                log_p_x = log_p_x.sum(dim=-1)
        else:
            log_p_x = None

        return {'log_p_x': log_p_x, 'log_p_z': log_p_z, 'x_params': x_params}

    def get_training_loss(self, x, posterior_flow=None, n_z=None):
        """SGVB training loss (negative ELBO)."""
        q_out = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
        z = q_out['z']
        log_q = q_out['log_q']

        p_out = self.model_net(z, x=x, n_z=n_z)
        log_p_x = p_out['log_p_x']
        log_p_z = p_out['log_p_z']

        if n_z is not None and z.dim() == 4:
            elbo = (log_p_x + log_p_z - log_q).mean(dim=0)
        else:
            elbo = log_p_x + log_p_z - log_q

        loss = -elbo.mean()
        return loss

    def get_reconstruction_log_prob(self, x, n_z=None, posterior_flow=None,
                                    last_point_only=True, per_dim=False):
        """Reconstruction log probability for anomaly scoring."""
        q_out = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
        z = q_out['z']

        if z.dim() == 4:
            z_mean = z.mean(dim=0)
            if n_z > 1:
                z_std = z.std(dim=0)
            else:
                z_std = torch.zeros_like(z_mean)
            z_info = torch.cat([z_mean, z_std], dim=-1)
        else:
            z_info = torch.cat([z, torch.zeros_like(z)], dim=-1)

        x_params = self.h_for_p_x(z if z.dim() == 3 else z.mean(dim=0))
        group_ndims = 0 if per_dim else 1
        mean = x_params['mean']
        std = x_params['std']

        if z.dim() == 4:
            mean = mean.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
            std = std.unsqueeze(0).expand(z.shape[0], -1, -1, -1)
            log_prob = gaussian_log_prob(x.unsqueeze(0), mean, std, group_ndims=group_ndims)
            log_prob = log_prob.mean(dim=0)
        else:
            log_prob = gaussian_log_prob(x, mean, std, group_ndims=group_ndims)

        if last_point_only:
            log_prob = log_prob[:, -1]

        return log_prob, z_info
