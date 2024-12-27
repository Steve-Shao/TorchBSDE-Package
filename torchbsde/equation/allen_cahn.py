import numpy as np
import torch

from .base import Equation


class AllenCahn(Equation):
    """Allen-Cahn equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, eqn_config, device=None, dtype=None):
        super(AllenCahn, self).__init__(eqn_config, device=device, dtype=dtype)
        self.x_init = np.zeros(self.dim)
        # self.x_init = np.zeros(self.dim) + np.random.normal(0, 1.0, self.dim)
        self.sigma = np.sqrt(2.0)

    def sample(self, num_sample):
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        return y - torch.pow(y, 3)

    def g_torch(self, t, x):
        return 0.5 / (1 + 0.2 * torch.sum(torch.square(x), 1, keepdim=True))