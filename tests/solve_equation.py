"""
The main file to run BSDE solver to solve parabolic partial differential equations (PDEs).
BSDE stands for Backward Stochastic Differential Equations, which are used to solve high-dimensional PDEs.
"""

# Standard library imports
import json
import os
import logging

# Third party imports
from absl import app                     # Provides core functionality for command line applications
from absl import flags                   # Handles command line flag definitions and parsing
from absl import logging as absl_logging # Provides logging functionality with additional features
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
    'exp_name',
    'test', 
    """Name of the experiment run. Used as prefix for log files and output data."""
)

# Get reference to defined flags
FLAGS = flags.FLAGS

# Set log directory for storing experiment outputs
FLAGS.log_dir = './logs'  # Directory for event logs and output arrays

def main(argv):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Ignore command line arguments as they're handled by FLAGS
    del argv

    # Load configuration file
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)

    # Set device and dtype
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    dtype = getattr(torch, config.get('dtype', 'float32'))
    print(f"Using device: {device}, dtype: {dtype}")
        
    # Initialize BSDE equation based on config
    bsde = getattr(eqn, config['eqn_config']['eqn_name'])(config['eqn_config'], device=device, dtype=dtype)

    # Create log directory if it doesn't exist
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)

    # Save configuration to JSON file
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(config, outfile, indent=2)

    # Configure logging format and level
    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')

    # Initialize and train BSDE solver
    logging.info('Begin to solve %s ' % config['eqn_config']['eqn_name'])
    bsde_solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    training_history = bsde_solver.train()

    # Log results if initial value exists
    if bsde.y_init:
        logging.info('Y0_true: %.4e' % bsde.y_init)
        logging.info('relative error of Y0: %s',
                    '{:.2%}'.format(abs(bsde.y_init - training_history[-1, 2])/bsde.y_init))

    # Save training history to CSV file
    np.savetxt('{}_training_history.csv'.format(path_prefix),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header='step,loss_function,target_value,elapsed_time',
               comments='')


if __name__ == '__main__':
    app.run(main)
