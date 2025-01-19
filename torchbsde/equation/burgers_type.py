import torch

from .base import Equation


class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config, device=None, dtype=None):
        super(BurgersType, self).__init__(eqn_config, device=device, dtype=dtype)
        self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        # self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype) + torch.randn(self.dim, device=self.device, dtype=self.dtype)
        self.y_init = 1 - 1.0 / (1 + torch.exp(torch.tensor(0.0, device=self.device, dtype=self.dtype) + torch.sum(self.x_init) / self.dim))
        self.sigma = float(self.dim)

    def sample(self, num_sample):
        with torch.no_grad():
            dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * self.sqrt_delta_t
            x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device, dtype=self.dtype)
            x_sample[:, :, 0] = self.x_init.expand(num_sample, self.dim)
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
            return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        return (y - (2 + self.dim) / 2.0 / self.dim) * torch.sum(z * self.sigma, dim=1, keepdim=True)

    def g_torch(self, t, x):
        return 1 - 1.0 / (1 + torch.exp(t + torch.sum(x, dim=1, keepdim=True) / self.dim))