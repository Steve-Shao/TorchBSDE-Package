import numpy as np
import torch

from .base import Equation


class HJBLQ(Equation):
    """Hamilton-Jacobi-Bellman equation with Linear-Quadratic control problem.
    
    This class implements the HJB equation described in the PNAS paper:
    "Solving high-dimensional partial differential equations using deep learning"
    doi.org/10.1073/pnas.1718942115
    
    Attributes:
        x_init (ndarray): Initial state vector, initialized as zero vector
        sigma (float): Diffusion coefficient, set to sqrt(2)
        lambd (float): Control cost coefficient, set to 1.0
    """
    def __init__(self, eqn_config):
        """Initialize the HJBLQ equation with given configuration.
        
        Args:
            eqn_config: Configuration dictionary containing PDE parameters
        """
        super(HJBLQ, self).__init__(eqn_config)
        # Initialize model parameters
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        """Generate sample paths for the forward SDE.
        
        Simulates the controlled diffusion process:
        dX_t = σ dW_t
        
        Args:
            num_sample (int): Number of Monte Carlo samples to generate
            
        Returns:
            tuple: (dw_sample, x_sample)
                - dw_sample: Brownian increments
                - x_sample: Sample paths of the state process
        """
        # Generate Brownian increments
        dw_sample = np.random.normal(size=[num_sample, self.dim, self.num_time_interval]) * self.sqrt_delta_t
        
        # Initialize state trajectory array
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        
        # Generate forward paths using Euler-Maruyama scheme
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
            
        return dw_sample, x_sample

    def f_torch(self, t, x, y, z):
        """Implements the driver function (drift term) of the BSDE.
        
        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state
            y (torch.Tensor): Current solution value
            z (torch.Tensor): Current gradient of the solution
            
        Returns:
            torch.Tensor: Value of the driver function
        """
        # Hamiltonian term: -λ|z|²/2
        return -self.lambd * torch.sum(torch.square(z), dim=1, keepdim=True) / 2

    def g_torch(self, t, x):
        """Implements the terminal condition of the PDE.
        
        Args:
            t (torch.Tensor): Terminal time
            x (torch.Tensor): Terminal state
            
        Returns:
            torch.Tensor: Value of the terminal condition
        """
        # Terminal cost: log((1 + |x|²)/2)
        return torch.log((1 + torch.sum(torch.square(x), dim=1, keepdim=True)) / 2)


if __name__ == "__main__":
    # -------------------------------------------------------
    # Test Configuration
    # -------------------------------------------------------
    config = {
        'dim': 3,
        'total_time': 1.0,
        'num_time_interval': 100
    }

    # Initialize equation
    equation = HJBLQ(config)
    print("\nInitialized HJBLQ equation with:")
    print(f"Dimension: {equation.dim}")
    print(f"Total time: {equation.total_time}")
    print(f"Time intervals: {equation.num_time_interval}")
    print(f"Delta t: {equation.delta_t}")
    print(f"Sigma: {equation.sigma}")
    print(f"Lambda: {equation.lambd}")

    # -------------------------------------------------------
    # Test Sample Generation
    # -------------------------------------------------------
    print("\nTesting sample generation:")
    num_test_samples = 5
    dw, x = equation.sample(num_test_samples)
    print(f"Brownian increments shape: {dw.shape}")
    print(f"State trajectories shape: {x.shape}")
    print(f"Initial state: {x[0,:,0]}")
    print(f"Final state: {x[0,:,-1]}")

    # -------------------------------------------------------
    # Test Driver Function
    # -------------------------------------------------------
    print("\nTesting driver function:")
    test_t = torch.tensor(0.5)
    test_x = torch.randn(num_test_samples, equation.dim)
    test_y = torch.randn(num_test_samples, 1)
    test_z = torch.randn(num_test_samples, equation.dim)
    
    f_value = equation.f_torch(test_t, test_x, test_y, test_z)
    print(f"Input shapes - t: {test_t.shape}, x: {test_x.shape}, y: {test_y.shape}, z: {test_z.shape}")
    print(f"Driver function output shape: {f_value.shape}")
    print(f"Sample driver value: {f_value[0].item()}")

    # -------------------------------------------------------
    # Test Terminal Condition
    # -------------------------------------------------------
    print("\nTesting terminal condition:")
    test_t_final = torch.tensor(equation.total_time)
    test_x_final = torch.randn(num_test_samples, equation.dim)
    
    g_value = equation.g_torch(test_t_final, test_x_final)
    print(f"Input shapes - t: {test_t_final.shape}, x: {test_x_final.shape}")
    print(f"Terminal condition output shape: {g_value.shape}")
    print(f"Sample terminal value: {g_value[0].item()}")

    # -------------------------------------------------------
    # Test Gradient Computation
    # -------------------------------------------------------
    print("\nTesting gradient computation:")
    test_x_final.requires_grad_(True)
    g_value = equation.g_torch(test_t_final, test_x_final)
    g_value.backward(torch.ones_like(g_value))
    grad = test_x_final.grad
    print(f"Gradient shape: {grad.shape}")
    print(f"Sample gradient: {grad[0].detach().numpy()}")