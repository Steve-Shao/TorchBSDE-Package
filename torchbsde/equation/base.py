import numpy as np
import torch


class Equation(object):
    """Base class for defining Partial Differential Equation (PDE) related functions.
    
    This class provides the foundation for implementing specific types of PDEs and their
    associated functions needed for solving Backward Stochastic Differential Equations (BSDEs).
    
    Attributes:
        dim (int): Dimension of the PDE problem
        total_time (float): Total time horizon for the equation
        num_time_interval (int): Number of time discretization intervals
        delta_t (float): Size of each time step (total_time / num_time_interval)
        sqrt_delta_t (float): Square root of delta_t, used in stochastic calculations
        y_init (float): Initial value of y, typically set by derived classes
    """

    def __init__(self, eqn_config):
        """Initialize the base equation with configuration parameters.
        
        Args:
            eqn_config: Configuration object containing PDE parameters including:
                - dim: Dimension of the problem
                - total_time: Total time horizon
                - num_time_interval: Number of time discretization steps
        """
        # Core problem dimensions and time parameters
        self.dim = eqn_config['dim']
        self.total_time = eqn_config['total_time']
        self.num_time_interval = eqn_config['num_time_interval']
        
        # Derived time step calculations
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        
        # Initial value placeholder
        self.y_init = None

    def sample(self, num_sample):
        """Sample forward Stochastic Differential Equation (SDE).
        
        Args:
            num_sample (int): Number of Monte Carlo samples to generate
            
        Raises:
            NotImplementedError: Must be implemented by derived classes
        """
        raise NotImplementedError

    def f_torch(self, t, x, y, z):
        """Generator function in the PDE (drift term).
        
        Args:
            t (torch.Tensor): Current time point
            x (torch.Tensor): Spatial position
            y (torch.Tensor): Solution value
            z (torch.Tensor): Gradient of the solution
            
        Raises:
            NotImplementedError: Must be implemented by derived classes
        """
        raise NotImplementedError

    def g_torch(self, t, x):
        """Terminal condition of the PDE (boundary condition at final time).
        
        Args:
            t (torch.Tensor): Time point (typically terminal time)
            x (torch.Tensor): Spatial position
            
        Raises:
            NotImplementedError: Must be implemented by derived classes
        """
        raise NotImplementedError


if __name__ == '__main__':
    # -------------------------------------------------------
    # Setup and Configuration
    # -------------------------------------------------------
    import json
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create test configuration
    config = json.loads('''{
        "dim": 10,
        "total_time": 1.0,
        "num_time_interval": 20
    }''')

    # Initialize base equation
    equation = Equation(config)

    # -------------------------------------------------------
    # Test Core Attributes
    # -------------------------------------------------------
    print("\nTesting core attributes:")
    print(f"Dimension: {equation.dim}")
    print(f"Total time: {equation.total_time}")
    print(f"Number of time intervals: {equation.num_time_interval}")
    print(f"Delta t: {equation.delta_t}")
    print(f"Sqrt delta t: {equation.sqrt_delta_t}")
    print(f"Initial y value: {equation.y_init}")

    # -------------------------------------------------------
    # Test Abstract Methods
    # -------------------------------------------------------
    print("\nTesting abstract methods:")
    try:
        equation.sample(10)
    except NotImplementedError:
        print("sample() correctly raises NotImplementedError")

    try:
        equation.f_torch(
            torch.tensor(0.0), 
            torch.zeros(1, equation.dim),
            torch.zeros(1, 1),
            torch.zeros(1, equation.dim)
        )
    except NotImplementedError:
        print("f_torch() correctly raises NotImplementedError")

    try:
        equation.g_torch(
            torch.tensor(1.0),
            torch.zeros(1, equation.dim)
        )
    except NotImplementedError:
        print("g_torch() correctly raises NotImplementedError")

    # -------------------------------------------------------
    # Test Tensor Types and Shapes
    # -------------------------------------------------------
    print("\nTesting tensor types and shapes:")
    test_t = torch.tensor(0.5)
    test_x = torch.randn(5, equation.dim)
    test_y = torch.randn(5, 1)
    test_z = torch.randn(5, equation.dim)

    print(f"Time tensor dtype: {test_t.dtype}")
    print(f"State tensor shape: {test_x.shape}")
    print(f"Value tensor shape: {test_y.shape}")
    print(f"Gradient tensor shape: {test_z.shape}")

    # -------------------------------------------------------
    # Test Numerical Properties
    # -------------------------------------------------------
    print("\nTesting numerical properties:")
    print(f"Delta t * num_intervals = total_time: {equation.delta_t * equation.num_time_interval == equation.total_time}")
    print(f"sqrt_delta_t^2 ≈ delta_t: {abs(equation.sqrt_delta_t**2 - equation.delta_t) < 1e-10}")