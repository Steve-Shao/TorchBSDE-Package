import numpy as np
import torch

from .base import Equation


class ReactionDiffusion(Equation):
    """
    Time-dependent reaction-diffusion-type example PDE in Section 4.7 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config, device=None, dtype=None):
        super(ReactionDiffusion, self).__init__(eqn_config, device=device, dtype=dtype)
        self._kappa = 0.6
        self.lambd = 1 / np.sqrt(self.dim)
        self.x_init = np.zeros(self.dim)
        self.y_init = 1 + self._kappa + np.sin(self.lambd * np.sum(self.x_init)) * np.exp(
            -self.lambd * self.lambd * self.dim * self.total_time / 2)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        exp_term = torch.exp((self.lambd ** 2) * self.dim * (t - self.total_time) / 2)
        sin_term = torch.sin(self.lambd * torch.sum(x, dim=1, keepdim=True))
        temp = y - self._kappa - 1 - sin_term * exp_term
        return torch.minimum(torch.tensor(1.0, device=self.device, dtype=self.dtype), torch.square(temp))

    def g_torch(self, t, x):
        return 1 + self._kappa + torch.sin(self.lambd * torch.sum(x, dim=1, keepdim=True))