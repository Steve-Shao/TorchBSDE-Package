"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).
BSDE stands for Backward Stochastic Differential Equations, which are used to solve high-dimensional PDEs.
"""

# Standard library imports
import os
import json  # Added for loading JSON config

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
    'config_path',
    'configs/hjb_lq_d100.json', 
    """Path to the JSON configuration file containing model and equation parameters."""
)
flags.DEFINE_string(
    'scenario_name',
    'new_test', 
    """Name of the experiment run. Used as prefix for log files and output data."""
)
# Get reference to defined flags
FLAGS = flags.FLAGS


def main(argv):
    # Ignore command line arguments as they're handled by FLAGS
    del argv

    # Load configuration file
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)

    # Add additional configurations
    config.update({
        "dtype": "float64",
        "test_folder_path": os.path.dirname(os.path.relpath(__file__)), 
        "test_scenario_name": FLAGS.scenario_name,
        "timezone": "America/Chicago", 
    })

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
    # Initialize and train BSDE solver
    bsde_solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    bsde_solver.train()
    bsde_solver.save_results()
    bsde_solver.plot_y0_history()
    bsde_solver.plot_training_history()
    bsde_solver.model.plot_subnet_gradients()


if __name__ == '__main__':
    app.run(main)
