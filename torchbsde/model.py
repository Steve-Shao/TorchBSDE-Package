"""
Implementation of the NonsharedModel class for solving BSDEs using neural networks.

This module contains the main model architecture that combines multiple feed-forward 
subnets to approximate solutions of high-dimensional BSDEs (Backward Stochastic 
Differential Equations).
"""

# Standard library imports
import json
import logging
import numpy as np
import torch
import torch.nn as nn

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
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32
        
        # ========== THIS PREVIOUS IMPLEMENTATION ASSUMES X0 IS CONSTANT ========== #

        # # Initialize y and z variables with random values
        # self.y_init = nn.Parameter(torch.FloatTensor(1).uniform_(
        #     self.net_config['y_init_range'][0],
        #     self.net_config['y_init_range'][1]
        # ).to(device=self.device, dtype=self.dtype))
        # self.z_init = nn.Parameter(torch.FloatTensor(1, self.eqn_config['dim']).uniform_(-0.1, 0.1).to(device=self.device, dtype=self.dtype))

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
        # time_stamp = torch.arange(0, self.eqn_config['num_time_interval'], device=self.device, dtype=self.dtype) * self.bsde.delta_t
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
        time_stamp = torch.arange(0, self.eqn_config['num_time_interval'], device=self.device, dtype=self.dtype) * self.bsde.delta_t
        y = self.y_init(x[:, :, 0], training) # / self.bsde.dim
        z = self.subnet[0](x[:, :, 0], training) / self.bsde.dim

        # Forward propagation through time steps
        for t in range(0, self.bsde.num_time_interval):
            z = self.subnet[t](x[:, :, t], training) / self.bsde.dim
            y = y - self.bsde.delta_t * (
                self.bsde.f_torch(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdim=True)

        # WHAT IS THIS `/ self.bsde.dim` DOING?
        # WHAT IS THIS `/ self.bsde.dim` DOING?
        # WHAT IS THIS `/ self.bsde.dim` DOING?

        return y


if __name__ == '__main__':
    # -------------------------------------------------------
    # Setup and Configuration
    # -------------------------------------------------------
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and parse configuration file
    config = json.loads('''{
        "eqn_config": {
            "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
            "eqn_name": "HJBLQ", 
            "total_time": 1.0,
            "dim": 100,
            "num_time_interval": 20
        },
        "net_config": {
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
    bsde = getattr(eqn, config['eqn_config']['eqn_name'])(config['eqn_config'], device=device, dtype=dtype)

    # Initialize NonsharedModel
    model = NonsharedModel(config, bsde, device=device, dtype=dtype)

    # -------------------------------------------------------
    # Model Structure Analysis
    # -------------------------------------------------------
    print("\nNonsharedModel Structure:")
    print("------------------------")
    # Create dummy input to build the model
    batch_size = 64
    dw = torch.zeros((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype)
    x = torch.zeros((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype)
    model((dw, x), training=False)  # Build model
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
    print(f"Y initialization range: {config['net_config']['y_init_range']}")
    print(f"Z initialization range: [-0.1, 0.1]")
    
    # Test each subnet individually
    test_input = torch.randn((batch_size, config['eqn_config']['dim']), device=device, dtype=dtype)
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
    y_zero = model((dw, x), training=False)
    print("\nTest with zero inputs:")
    print(f"Output shape: {y_zero.shape}")
    print(f"Output mean: {torch.mean(y_zero):.6f}")
    print(f"Output std: {torch.std(y_zero):.6f}")
    print(f"Output min: {torch.min(y_zero):.6f}")
    print(f"Output max: {torch.max(y_zero):.6f}")

    # Test 2: Random normal inputs
    dw_random = torch.randn((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype)
    x_random = torch.randn((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype)
    y_random = model((dw_random, x_random), training=False)
    print("\nTest with random normal inputs:")
    print(f"Output shape: {y_random.shape}")
    print(f"Output mean: {torch.mean(y_random):.6f}")
    print(f"Output std: {torch.std(y_random):.6f}")
    print(f"Output min: {torch.min(y_random):.6f}")
    print(f"Output max: {torch.max(y_random):.6f}")

    # Test 3: Edge case with large values
    dw_large = torch.randn((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype) * 10
    x_large = torch.randn((batch_size, config['eqn_config']['dim'], config['eqn_config']['num_time_interval']), device=device, dtype=dtype) * 10
    y_large = model((dw_large, x_large), training=False)
    print("\nTest with large inputs:")
    print(f"Output shape: {y_large.shape}")
    print(f"Output mean: {torch.mean(y_large):.6f}")
    print(f"Output std: {torch.std(y_large):.6f}")
    print(f"Output min: {torch.min(y_large):.6f}")
    print(f"Output max: {torch.max(y_large):.6f}")
