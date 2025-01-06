"""
Implementation of the NonsharedModel class for solving BSDEs using neural networks.

This module contains the main model architecture that combines multiple feed-forward 
subnets to approximate solutions of high-dimensional BSDEs (Backward Stochastic 
Differential Equations).
"""

# Standard library imports
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Local imports
from . import equation as eqn
from .subnet import FeedForwardSubNet


class NonsharedModel(nn.Module):
    """Neural network model that uses separate subnets for each time step.
    
    This model implements a deep BSDE solver where each time step has its own 
    subnet for better approximation capability.
    
    Args:
        config: Configuration object containing model and equation parameters
        bsde: BSDE equation object defining the problem to solve
        device: Device to run computations on
        dtype: Data type for tensors
    """
    def __init__(self, config, bsde, device=None, dtype=None):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.equation_config = config['equation_config']
        self.network_config = config['network_config']
        self.bsde = bsde
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32
        
        # Paths for experiment directory
        self.test_folder_path = self.config.get('test_folder_path', 'tests/')
        self.test_scenario_name = self.config.get('test_scenario_name', 'new_test')
        self.exp_dir = None  # Will be set in _create_experiment_directory or loaded
        # Create experiment directory
        self._create_experiment_directory()

        # Get sigma from bsde or set to 1 if not found
        self.sigma = getattr(self.bsde, 'sigma', 1)
        # Reshape scalar sigma into diagonal matrix if it's a scalar and repeat for each time step
        if isinstance(self.sigma, (int, float, np.floating, np.integer)):
            diag_matrix = self.sigma * torch.eye(self.equation_config['dim'], device=self.device, dtype=self.dtype)
            self.sigma = diag_matrix.unsqueeze(-1).repeat(1, 1, self.bsde.num_time_interval)
        # If sigma is [dim, num_time_interval], expand to [dim, dim, num_time_interval] as diagonal matrices
        elif self.sigma.ndim == 2 and self.sigma.shape[0] == self.equation_config['dim']:
            diag_values = self.sigma  # [dim, num_time_interval]
            dim = self.equation_config['dim']
            expanded_sigma = torch.zeros(dim, dim, self.bsde.num_time_interval, device=self.device, dtype=self.dtype)
            for t in range(self.bsde.num_time_interval):
                expanded_sigma[:,:,t] = torch.diag(diag_values[:,t])
            self.sigma = expanded_sigma
                
        # ========== THIS PREVIOUS IMPLEMENTATION ASSUMES X0 IS CONSTANT ========== #

        # # Initialize y and z variables with random values
        # self.y_init = nn.Parameter(torch.FloatTensor(1).uniform_(
        #     self.network_config['y_init_range'][0],
        #     self.network_config['y_init_range'][1]
        # ).to(device=self.device, dtype=self.dtype))
        # self.z_init = nn.Parameter(torch.FloatTensor(1, self.equation_config['dim']).uniform_(-0.1, 0.1).to(device=self.device, dtype=self.dtype))

        # # Create subnet for each time step except the last one
        # self.subnet = nn.ModuleList([
        #     FeedForwardSubNet(config, device=self.device, dtype=self.dtype) 
        #     for _ in range(self.bsde.num_time_interval-1)
        # ])

        # ========== WE NOW ALLOW X0 TO BE RANDOMIZED ========== #

        # Initialize y 
        self.y_init = FeedForwardSubNet(config, device=self.device, dtype=self.dtype, is_derivative=False) 

        # Create subnet for each time step except the last one
        self.subnet = nn.ModuleList([
            FeedForwardSubNet(config, device=self.device, dtype=self.dtype) 
            for _ in range(self.bsde.num_time_interval)
        ])

    def _create_experiment_directory(self):
        """
        Creates the experiment directory if it doesn't exist. 
        This directory will store config, logs, model checkpoints, etc.
        """
        self.exp_dir = os.path.join(self.test_folder_path, self.test_scenario_name)
        os.makedirs(self.exp_dir, exist_ok=True)

    def calculate_negative_loss(self, func):
        """Calculate the negative loss for a given tensor.
        
        Args:
            func: Tensor to calculate negative loss from
        
        Returns:
            negative_loss: Scalar tensor representing the negative loss
        """
        # zero_func = torch.minimum(torch.min(func, dim=1, keepdim=True)[0], torch.tensor(0.0, device=func.device, dtype=func.dtype))
        zero_func = torch.minimum(torch.min(func, dim=0, keepdim=True)[0], torch.tensor(0.0, device=func.device, dtype=func.dtype))
        negative_loss = torch.sum(zero_func ** 2)
        return negative_loss

    def forward(self, inputs, training):
        """Forward pass of the model.
        
        Args:
            inputs: Tuple of (dw, x) where:
                dw: Brownian increments tensor
                x: State process tensor
            training: Boolean indicating training vs inference mode
            
        Returns:
            y: Terminal value approximation
        """

        # ========== THIS PREVIOUS IMPLEMENTATION ASSUMES X0 IS CONSTANT ========== #

        # dw, x = inputs
        # time_stamp = torch.arange(0, self.equation_config['num_time_interval'], device=self.device, dtype=self.dtype) * self.bsde.delta_t
        # all_one_vec = torch.ones(dw.shape[0], 1, device=self.device, dtype=self.dtype)
        # y = torch.matmul(all_one_vec, self.y_init)
        # z = torch.matmul(all_one_vec, self.z_init)

        # # Forward propagation through time steps
        # for t in range(0, self.bsde.num_time_interval-1):
        #     y = y - self.bsde.delta_t * (
        #         self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z)
        #     ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)
        #     z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
            
        # # Handle terminal time step
        # y = y - self.bsde.delta_t * (
        #     self.bsde.f_torch(time_stamp[-1], x[:, :, -2], y, z) 
        # ) + torch.sum(z * dw[:, :, -1], 1, keepdim=True)

        # ========== WE NOW ALLOW X0 TO BE RANDOMIZED ========== #

        dw, x = inputs
        time_stamp = torch.arange(0, self.equation_config['num_time_interval'], device=self.device, dtype=self.dtype) * self.bsde.delta_t
        y = self.y_init(x[:, :, 0], training) # / self.bsde.dim # (this is an enginnering trick used my Han et al.)
        negative_loss = self.calculate_negative_loss(y)

        # Forward propagation through time steps
        for t in range(0, self.bsde.num_time_interval):
            z = self.subnet[t](x[:, :, t], training) # / self.bsde.dim # (this is an enginnering trick used my Han et al.)
            negative_loss += self.calculate_negative_loss(z)

            y = y - self.bsde.delta_t * (
                self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(torch.matmul(z, self.sigma[:, :, t]) * dw[:, :, t], 1, keepdim=True)

        return y, negative_loss

    def plot_subnet_gradients(self, filename='subnet_gradients.png'):
        """
        Plots sample paths of the gradient subnet outputs over time.
        
        Args:
            filename (str): Filename to save the plot in the experiment directory.
        """
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            # Generate sample input data from bsde
            num_samples = 20
            sample_data = self.bsde.sample(num_samples)
            sample_dw, sample_x = sample_data
            sample_dw = torch.tensor(sample_dw, dtype=self.dtype, device=self.device)
            sample_x = torch.tensor(sample_x, dtype=self.dtype, device=self.device)

            # Initialize a list to store subnet outputs for each dimension
            subnet_outputs = [[] for _ in range(self.bsde.dim)]

            # Iterate over each time step and collect subnet outputs
            for t in range(self.subnet.__len__()):
                input_x_t = sample_x[:, :, t]
                output_z = self.subnet[t](input_x_t, training=False)  # Shape: (num_samples, dim)
                output_z = output_z.cpu().numpy()
                for dim_idx in range(self.bsde.dim):
                    subnet_outputs[dim_idx].append(output_z[:, dim_idx])

            # Time steps for plotting
            time_steps = np.linspace(0, self.bsde.total_time,
                                   self.bsde.num_time_interval)

            # Create plot
            fig, axes = plt.subplots(self.bsde.dim, 1, figsize=(8, 1.5 * self.bsde.dim), sharex=True)

            if self.bsde.dim == 1:
                axes = [axes]

            for dim_idx in range(self.bsde.dim):
                for sample_idx in range(num_samples):
                    axes[dim_idx].plot(time_steps, subnet_outputs[dim_idx][sample_idx], alpha=0.6)
                axes[dim_idx].set_ylabel(f'Dimension {dim_idx + 1}')
                axes[dim_idx].grid(True)

            axes[-1].set_xlabel('Time')
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.exp_dir, filename)
            plt.savefig(plot_path)
            plt.close()


if __name__ == '__main__':
    # -------------------------------------------------------
    # Setup and Configuration
    # -------------------------------------------------------
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and parse configuration file
    config = json.loads('''{
        "equation_config": {
            "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
            "eqn_name": "HJBLQ", 
            "total_time": 1.0,
            "dim": 100,
            "num_time_interval": 20
        },
        "network_config": {
            "y_init_range": [0, 1],
            "num_hiddens": [110, 110],
            "lr_values": [1e-2, 1e-2],
            "lr_boundaries": [1000],
            "num_iterations": 2000,
            "batch_size": 64,
            "valid_size": 256,
            "logging_frequency": 100,
            "verbose": true
        }
    }''')
    # Initialize BSDE equation based on config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    bsde = getattr(eqn, config['equation_config']['eqn_name'])(config['equation_config'], device=device, dtype=dtype)

    # Initialize NonsharedModel
    model = NonsharedModel(config, bsde, device=device, dtype=dtype)

    # -------------------------------------------------------
    # Model Structure Analysis
    # -------------------------------------------------------
    print("\nNonsharedModel Structure:")
    print("------------------------")
    # Create dummy input to build the model
    batch_size = 64
    dw = torch.zeros((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype)
    x = torch.zeros((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype)
    y, _ = model((dw, x), training=False)  # Build model
    print(model)

    # Print model parameters
    print("\nModel Parameters:")
    print("----------------")
    total_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        print(f"{name}: shape={param.shape}, params={params}")
    print(f"Total trainable parameters: {total_params}")

    # -------------------------------------------------------
    # Subnet Testing
    # -------------------------------------------------------
    print("\nSubnet Details and Tests:")
    print("-----------------------")
    print(f"Number of subnets: {len(model.subnet)}")
    print(f"Y initialization range: {config['network_config']['y_init_range']}")
    print(f"Z initialization range: [-0.1, 0.1]")
    
    # Test each subnet individually
    test_input = torch.randn((batch_size, config['equation_config']['dim']), device=device, dtype=dtype)
    for i, subnet in enumerate(model.subnet):
        subnet_output = subnet(test_input, training=False)
        print(f"\nSubnet {i} test:")
        print(f"Output shape: {subnet_output.shape}")
        print(f"Output mean: {torch.mean(subnet_output):.6f}")
        print(f"Output std: {torch.std(subnet_output):.6f}")
        print(f"Output min: {torch.min(subnet_output):.6f}")
        print(f"Output max: {torch.max(subnet_output):.6f}")

    # -------------------------------------------------------
    # Full Model Forward Pass Tests
    # -------------------------------------------------------
    print("\nFull Model Forward Pass Tests:")
    print("----------------------------")
    
    # Test 1: Zero inputs
    y_zero, _ = model((dw, x), training=False)
    print("\nTest with zero inputs:")
    print(f"Output shape: {y_zero.shape}")
    print(f"Output mean: {torch.mean(y_zero):.6f}")
    print(f"Output std: {torch.std(y_zero):.6f}")
    print(f"Output min: {torch.min(y_zero):.6f}")
    print(f"Output max: {torch.max(y_zero):.6f}")

    # Test 2: Random normal inputs
    dw_random = torch.randn((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype)
    x_random = torch.randn((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype)
    y_random, _ = model((dw_random, x_random), training=False)
    print("\nTest with random normal inputs:")
    print(f"Output shape: {y_random.shape}")
    print(f"Output mean: {torch.mean(y_random):.6f}")
    print(f"Output std: {torch.std(y_random):.6f}")
    print(f"Output min: {torch.min(y_random):.6f}")
    print(f"Output max: {torch.max(y_random):.6f}")

    # Test 3: Edge case with large values
    dw_large = torch.randn((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype) * 10
    x_large = torch.randn((batch_size, config['equation_config']['dim'], config['equation_config']['num_time_interval']), device=device, dtype=dtype) * 10
    y_large, _ = model((dw_large, x_large), training=False)
    print("\nTest with large inputs:")
    print(f"Output shape: {y_large.shape}")
    print(f"Output mean: {torch.mean(y_large):.6f}")
    print(f"Output std: {torch.std(y_large):.6f}")
    print(f"Output min: {torch.min(y_large):.6f}")
    print(f"Output max: {torch.max(y_large):.6f}")
