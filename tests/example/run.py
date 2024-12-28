"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).
BSDE stands for Backward Stochastic Differential Equations, which are used to solve high-dimensional PDEs.
"""

# Standard library imports
import os

# Third party imports
from absl import app                     # Provides core functionality for command line applications
from absl import flags                   # Handles command line flag definitions and parsing
import numpy as np
import torch

# Local imports
from torchbsde import equation as eqn
from torchbsde.solver import BSDESolver

# Define command line flags for configuration
flags.DEFINE_string(
    'scenario_name',
    'new_scenario', 
    """Name of the experiment run. Used as prefix for log files and output data."""
)
# Get reference to defined flags
FLAGS = flags.FLAGS


def main(argv):
    # Ignore command line arguments as they're handled by FLAGS
    del argv

    # Set configuration
    config = {
        "equation_config": {
            "_comment": "HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115",
            "eqn_name": "HJBLQ", 
            "total_time": 1.0,
            "dim": 100,
            "num_time_interval": 20
        },
        "network_config": {
            "use_bn_input": True,
            "use_bn_hidden": True,
            "use_bn_output": True,
            "use_shallow_y_init": True,
            "num_hiddens": [110, 110]
        },
        "solver_config": {
            "batch_size": 64,
            "valid_size": 256,
            "lr_start_value": 1e-2,
            "lr_decay_rate": 0.5,
            "lr_boundaries": [1000, 2000, 3000],
            "num_iterations": 5000,
            "logging_frequency": 100,
            "negative_loss_penalty": 0.0,
            "delta_clip": 50,
            "verbose": True,
        },
        "dtype": "float64",
        "test_folder_path": os.path.dirname(os.path.relpath(__file__)), 
        "test_scenario_name": FLAGS.scenario_name,
        "timezone": "America/Chicago", 
    }

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device and dtype
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    dtype = getattr(torch, config.get('dtype', 'float32'))
        
    # Initialize BSDE equation based on config
    bsde = getattr(eqn, config['equation_config']['eqn_name'])(config['equation_config'], device=device, dtype=dtype)
    # # Initialize and train BSDE solver
    bsde_solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    bsde_solver.train()


if __name__ == '__main__':
    app.run(main)
