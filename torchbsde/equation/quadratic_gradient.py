import numpy as np
import torch

from .base import Equation


class QuadraticGradient(Equation):
    """
    An example PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config, device=None, dtype=None):
        super(QuadraticGradient, self).__init__(eqn_config, device=device, dtype=dtype)
        self.alpha = 0.4
        self.x_init = np.zeros(self.dim)
        base = self.total_time + np.sum(np.square(self.x_init) / self.dim)
        self.y_init = np.sin(np.power(base, self.alpha))

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        x_square = torch.sum(torch.square(x), dim=1, keepdim=True)
        base = self.total_time - t + x_square / self.dim
        base_alpha = torch.pow(base, self.alpha)
        derivative = self.alpha * torch.pow(base, self.alpha - 1) * torch.cos(base_alpha)
        term1 = torch.sum(torch.square(z), dim=1, keepdim=True)
        term2 = -4.0 * (derivative ** 2) * x_square / (self.dim ** 2)
        term3 = derivative
        term4 = -0.5 * (
            2.0 * derivative + 4.0 / (self.dim ** 2) * x_square * self.alpha * (
                (self.alpha - 1) * torch.pow(base, self.alpha - 2) * torch.cos(base_alpha) - (
                    self.alpha * torch.pow(base, 2 * self.alpha - 2) * torch.sin(base_alpha)
                    )
                )
            )
        return term1 + term2 + term3 + term4

    def g_torch(self, t, x):
        return torch.sin(
            torch.pow(torch.sum(torch.square(x), dim=1, keepdim=True) / self.dim, self.alpha))