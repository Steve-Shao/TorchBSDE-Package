import numpy as np
import torch

from .base import Equation


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config, device=None, dtype=None):
        super(AllenCahn, self).__init__(eqn_config, device=device, dtype=dtype)
        self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        # self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype) + torch.randn(self.dim, device=self.device, dtype=self.dtype)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        with torch.no_grad():
            dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * self.sqrt_delta_t
            x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device, dtype=self.dtype)
            x_sample[:, :, 0] = self.x_init.expand(num_sample, self.dim)
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
            return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_torch(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(torch.square(x), 1, keepdim=True))