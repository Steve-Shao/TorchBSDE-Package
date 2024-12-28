"""
Implementation of the BSDESolver class for training neural networks to solve BSDEs.

This module contains the main solver that handles training, loss computation, and optimization
of the neural network model for solving Backward Stochastic Differential Equations (BSDEs).
"""

import os
import time
import pytz  
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from . import equation as eqn
from .model import NonsharedModel


class BSDESolver(object):
    """Solver class that trains neural networks to solve BSDEs.
    
    This class handles the training loop, loss computation, and optimization of the neural
    network model. It uses stochastic gradient descent with the Adam optimizer.

    Args:
        config (dict): Configuration dictionary containing model and equation parameters
        bsde (object): BSDE equation object defining the problem to solve
        device (torch.device, optional): Device to run the computations on. Defaults to CUDA if available.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to float32.
    """
    def __init__(self, config, bsde, device=None, dtype=None):
        # Create experiment directory if it doesn't exist
        self.test_folder_path = config.get('test_folder_path', 'tests/')
        self.test_scenario_name = config.get('test_scenario_name', 'new_test')
        self.exp_dir = os.path.join(self.test_folder_path, self.test_scenario_name)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        # Initialize device and dtype
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32
        self.timezone = pytz.timezone(config.get('timezone', 'UTC'))

        # Setup logging with a new logger instance
        self.logger = logging.getLogger(self.test_scenario_name)  # Create a named logger
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Prevent logger from propagating messages to parent loggers
        self.logger.propagate = False
        
        # Create file handler with a fixed filename
        log_file = os.path.join(self.exp_dir, 'log.log')
        file_handler = logging.FileHandler(log_file, mode='a')  # 'a' means append mode
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter with time format up to seconds
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Print initialization information
        self.logger.info("========== Initialization Information ==========")
        self.logger.info(f"Date                 : {datetime.now(self.timezone).strftime('%Y-%m-%d')}")
        self.logger.info(f"Time                 : {datetime.now(self.timezone).strftime('%H:%M:%S')}")
        self.logger.info(f"Experiment directory : {self.exp_dir}")
        self.logger.info(f"Device               : {self.device}")
        self.logger.info(f"Dtype                : {self.dtype}")

        # Save current configuration to JSON file
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        self.equation_config = config['equation_config']
        self.network_config = config['network_config']
        self.solver_config = config['solver_config']
        self.bsde = bsde

        # Negative loss penalty
        self.negative_loss_penalty = self.solver_config.get('negative_loss_penalty', 0.0)
        # Constant for clipping the loss delta to avoid numerical instability
        self.delta_clip = self.solver_config.get('delta_clip', 50.0)

        # Initialize model and get reference to y_init parameter
        self.model = NonsharedModel(config, bsde, device=self.device, dtype=self.dtype)
        self.y_init = self.model.y_init

        # Setup optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.solver_config['lr_start_value'], eps=1e-8)
        if len(self.solver_config['lr_boundaries']) > 0:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.solver_config['lr_boundaries'], gamma=self.solver_config['lr_decay_rate']
            )
        else:
            self.lr_scheduler = None

    def train(self):
        """Trains the model using stochastic gradient descent.
        
        Performs training iterations, periodically evaluating on validation data
        and logging the results.

        Returns:
            numpy.ndarray: Training history containing step, training loss, validation loss, Y0, learning rate, and elapsed time
        """
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.solver_config['valid_size'])
        valid_dw, valid_x = valid_data
        valid_dw = torch.tensor(valid_dw, dtype=self.dtype, device=self.device)
        valid_x = torch.tensor(valid_x, dtype=self.dtype, device=self.device)

        # Initial validation
        with torch.no_grad():
            val_loss = self.loss_fn(valid_dw, valid_x, training=False).item()
            y_init = self.y_init(valid_x[:, :, 0], training=False).data.cpu().numpy().mean()
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            training_history.append([0, float('nan'), val_loss, y_init, current_lr, elapsed_time])
            if self.solver_config['verbose']:
                self.logger.info(f"step: {0:5}, val loss: {val_loss:.6f}, Y0: {y_init:.6f}, lr: {current_lr:.6f}, elapsed time: {int(elapsed_time)}")

        for step in range(1, self.solver_config['num_iterations'] + 1):
            # Sample training data
            train_data = self.bsde.sample(self.solver_config['batch_size'])
            dw, x = train_data
            dw = torch.tensor(dw, dtype=self.dtype, device=self.device)
            x = torch.tensor(x, dtype=self.dtype, device=self.device)

            # Perform a training step
            self.train_step(dw, x)

            # Update learning rate scheduler if applicable
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Log progress at specified frequency
            if step % self.solver_config['logging_frequency'] == 0:
                with torch.no_grad():
                    val_loss = self.loss_fn(valid_dw, valid_x, training=False).item()
                    y_init = self.y_init(valid_x[:, :, 0], training=False).data.cpu().numpy().mean()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed_time = time.time() - start_time
                    training_history.append([step, float('nan'), val_loss, y_init, current_lr, elapsed_time])
                    if self.solver_config['verbose']:
                        self.logger.info(f"step: {step:5}, val loss: {val_loss:.6f}, Y0: {y_init:.6f}, lr: {current_lr:.6f}, elapsed time: {int(elapsed_time)}")
            else:
                # Record training loss every iteration
                train_loss = self.loss_fn(dw, x, training=True).item()
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed_time = time.time() - start_time
                training_history.append([step, train_loss, float('nan'), y_init, current_lr, elapsed_time])

        # Convert training history to numpy array
        training_history = np.array(training_history)
        
        # Save training history to CSV file
        np.savetxt(os.path.join(self.exp_dir, 'training_history.csv'),
                  training_history,
                  fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%d'],
                  delimiter=",",
                  header='step,train_loss,val_loss,y0,learning_rate,elapsed_time',
                  comments='')
        
        # return training_history

    def loss_fn(self, dw, x, training):
        """Computes the loss for training the model.
        
        Calculates the mean squared error between predicted and true terminal values,
        with special handling for large deviations using linear approximation.

        Args:
            dw (torch.Tensor): Brownian increments
            x (torch.Tensor): State process
            training (bool): Indicates if in training mode

        Returns:
            torch.Tensor: Computed loss value
        """
        y_terminal, negative_loss = self.model((dw, x), training)
        g_terminal = self.bsde.g_torch(torch.tensor(self.bsde.total_time, dtype=self.dtype, device=self.device), x[:, :, -1])
        delta = y_terminal - g_terminal
        abs_delta = torch.abs(delta)
        mask = abs_delta < self.delta_clip
        loss = torch.mean(torch.where(mask, delta ** 2, 2 * self.delta_clip * abs_delta - self.delta_clip ** 2))
        loss += self.negative_loss_penalty * negative_loss
        return loss

    def grad(self, dw, x, training):
        """Computes gradients of the loss with respect to model parameters.
        
        Args:
            dw (torch.Tensor): Brownian increments
            x (torch.Tensor): State process
            training (bool): Indicates if in training mode

        Returns:
            torch.Tensor: Computed gradients
        """
        loss = self.loss_fn(dw, x, training)
        self.optimizer.zero_grad()
        loss.backward()
        grads = [param.grad for param in self.model.parameters()]
        return grads

    def train_step(self, dw, x):
        """Performs one training step using gradient descent.
        
        Args:
            dw (torch.Tensor): Brownian increments
            x (torch.Tensor): State process
        """
        # Compute gradients
        grads = self.grad(dw, x, training=True)
        
        # Update model parameters
        self.optimizer.step()


if __name__ == '__main__':
    # -------------------------------------------------------
    # Setup and Configuration
    # -------------------------------------------------------
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and parse configuration file
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
        "test_folder_path": 'tests/',
        "test_scenario_name": 'new_test',
        "timezone": "America/Chicago", 
    }

    # -------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------
    # Initialize BSDE equation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    bsde = getattr(eqn, config['equation_config']['eqn_name'])(config['equation_config'], device=device, dtype=dtype)

    # Initialize and train BSDE solver
    solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    
    # -------------------------------------------------------
    # Testing Model Components
    # -------------------------------------------------------
    # Test model initialization
    print("\nTesting model initialization...")
    print("Model architecture:", solver.model)
    print("Number of trainable variables:", len(list(solver.model.parameters())))
    
    # Test data generation
    print("\nTesting data generation...")
    test_batch = bsde.sample(config['solver_config']['batch_size'])
    test_dw, test_x = test_batch
    test_dw = torch.tensor(test_dw, dtype=solver.dtype, device=solver.device)
    test_x = torch.tensor(test_x, dtype=solver.dtype, device=solver.device)
    print("Sample batch shapes - dw:", test_dw.shape, "x:", test_x.shape)
    
    # Test forward pass
    print("\nTesting forward pass...")
    y_pred, negative_loss = solver.model((test_dw, test_x), training=False)
    print("Model output shape:", y_pred.shape)
    
    # Test loss computation
    print("\nTesting loss computation...")
    loss = solver.loss_fn(test_dw, test_x, training=False)
    print("Initial loss value:", loss.item())
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    grads = solver.grad(test_dw, test_x, training=True)
    print("Number of gradient tensors:", len(grads))
    print("\nGradient shapes:")
    for i, shape in enumerate(g.shape for g in grads):
        if i > 0 and i % 3 == 0:
            print()  # Add line break every 3 shapes for readability
        print(f"  {shape}", end="  ")
    print("\n")

    # -------------------------------------------------------
    # Training and Results
    # -------------------------------------------------------
    # Test full training
    print("\nStarting full training...")
    training_history = solver.train()
    
    # Output detailed training results
    print("\nTraining completed.")
    print("Training history shape:", training_history.shape)
    print("Columns: [step, training_loss, validation_loss, Y0, learning_rate, elapsed_time]")
    print("\nFinal metrics:")
    print("Final Loss:", training_history[-1, 1])
    print("Initial Y Value:", training_history[-1, 2]) 
    print("Total Elapsed Time:", training_history[-1, 3])
    print("Loss progression:", training_history[:, 1])
    
    # -------------------------------------------------------
    # Model Persistence Analysis
    # -------------------------------------------------------
    # Test model persistence capabilities
    print("\nModel persistence capabilities:")
    print("Trainable variables:")
    for name, param in solver.model.named_parameters():
        print(f"  {name}")
        
    print("\nVariable shapes:")
    for name, param in solver.model.named_parameters():
        print(f"  {name}: {param.shape}")
