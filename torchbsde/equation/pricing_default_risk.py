import torch

from .base import Equation


class PricingDefaultRisk(Equation):
    """
    Nonlinear Black-Scholes equation with default risk in PNAS paper
    doi.org/10.1073/pnas.1718942115
    """
    def __init__(self, eqn_config, device=None, dtype=None):
        super(PricingDefaultRisk, self).__init__(eqn_config, device=device, dtype=dtype)
        self.x_init = torch.ones(self.dim, device=self.device, dtype=self.dtype) * 100.0
        self.sigma = 0.2
        self.rate = 0.02   # interest rate R
        self.delta = 2.0 / 3
        self.gammah = 0.2
        self.gammal = 0.02
        self.mu_bar = 0.02
        self.vh = 50.0
        self.vl = 70.0
        self.slope = (self.gammah - self.gammal) / (self.vh - self.vl)

    def sample(self, num_sample):
        with torch.no_grad():
            dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * self.sqrt_delta_t
            x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device, dtype=self.dtype)
            x_sample[:, :, 0] = self.x_init.expand(num_sample, self.dim)
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = (1 + self.mu_bar * self.delta_t) * x_sample[:, :, i] + (
                    self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])
            return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        piecewise_linear = torch.relu(
            torch.relu(y - self.vh) * self.slope + self.gammah - self.gammal) + self.gammal
        return (-(1 - self.delta) * piecewise_linear - self.rate) * y

    def g_torch(self, t, x):
        return torch.min(x, dim=1, keepdim=True)[0]