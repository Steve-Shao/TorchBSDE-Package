import torch

from .base import Equation


class PricingDiffRate(Equation):
    """
    Nonlinear Black-Scholes equation with different interest rates for borrowing and lending
    in Section 4.4 of Comm. Math. Stat. paper doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config, device=None, dtype=None):
        super(PricingDiffRate, self).__init__(eqn_config, device=device, dtype=dtype)
        self.x_init = torch.ones(self.dim, device=self.device, dtype=self.dtype) * 100
        self.sigma = 0.2
        self.mu_bar = 0.06
        self.rl = 0.04
        self.rb = 0.06
        self.alpha = 1.0 / self.dim

    def sample(self, num_sample):
        with torch.no_grad():
            dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * self.sqrt_delta_t
            x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device, dtype=self.dtype)
            x_sample[:, :, 0] = self.x_init.expand(num_sample, self.dim)
            factor = torch.exp(torch.tensor(self.mu_bar - (self.sigma**2)/2, device=self.device, dtype=self.dtype) * self.delta_t)
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = (factor * torch.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
            return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        temp = torch.sum(z * self.sigma, dim=1, keepdim=True) / self.sigma
        return -self.rl * y - (self.mu_bar - self.rl) * temp + (
            (self.rb - self.rl) * torch.clamp(temp - y, min=0))

    def g_torch(self, t, x):
        temp = torch.max(x, dim=1, keepdim=True)[0]
        return torch.clamp(temp - 120, min=0) - 2 * torch.clamp(temp - 150, min=0)