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
    def __init__(self, eqn_config, device=None, dtype=None):
        """Initialize the HJBLQ equation with given configuration.
        
        Args:
            eqn_config: Configuration dictionary containing PDE parameters
            device: Device to run computations on
            dtype: Data type for tensors
        """
        super(HJBLQ, self).__init__(eqn_config, device=device, dtype=dtype)
        # Initialize model parameters
        self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        # self.x_init = torch.zeros(self.dim, device=self.device, dtype=self.dtype) + torch.randn(self.dim, device=self.device, dtype=self.dtype)
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
        with torch.no_grad():
            # Generate Brownian increments
            dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval, device=self.device, dtype=self.dtype) * self.sqrt_delta_t
            
            # Initialize state trajectory array
            x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1, device=self.device, dtype=self.dtype)
            x_sample[:, :, 0] = self.x_init.expand(num_sample, self.dim)
            
            # Generate forward paths using Euler-Maruyama scheme
            for i in range(self.num_time_interval):
                x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
                
            return dw_sample, x_sample, None

    def f_torch(self, t, x, y, z, u, step):
        """Implements the driver function (drift term) of the BSDE.
        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state
            y (torch.Tensor): Current solution value
            z (torch.Tensor): Current gradient of the solution
            u (torch.Tensor): Control process (optional, could be None)
            step (int): Current time step
        Returns:
            torch.Tensor: Value of the driver function
        """
        # Hamiltonian term: -λ|z|²/2
        return -self.lambd * torch.sum(torch.square(z * self.sigma), dim=1, keepdim=True) / 2

    def g_torch(self, t, x, step):
        """Implements the terminal condition of the PDE.
        Args:
            t (torch.Tensor): Terminal time
            x (torch.Tensor): Terminal state
            step (int): Current time step
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Initialize equation
    equation = HJBLQ(config, device=device, dtype=dtype)
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
    test_t = torch.tensor(0.5, device=device, dtype=dtype)
    test_x = torch.randn(num_test_samples, equation.dim, device=device, dtype=dtype)
    test_y = torch.randn(num_test_samples, 1, device=device, dtype=dtype)
    test_z = torch.randn(num_test_samples, equation.dim, device=device, dtype=dtype)
    
    f_value = equation.f_torch(test_t, test_x, test_y, test_z)
    print(f"Input shapes - t: {test_t.shape}, x: {test_x.shape}, y: {test_y.shape}, z: {test_z.shape}")
    print(f"Driver function output shape: {f_value.shape}")
    print(f"Sample driver value: {f_value[0].item()}")

    # -------------------------------------------------------
    # Test Terminal Condition
    # -------------------------------------------------------
    print("\nTesting terminal condition:")
    test_t_final = torch.tensor(equation.total_time, device=device, dtype=dtype)
    test_x_final = torch.randn(num_test_samples, equation.dim, device=device, dtype=dtype)
    
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
    print(f"Sample gradient: {grad[0].detach().cpu().numpy()}")
