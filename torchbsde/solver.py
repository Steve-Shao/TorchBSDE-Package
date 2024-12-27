"""
Implementation of the BSDESolver class for training neural networks to solve BSDEs.

This module contains the main solver that handles training, loss computation, and optimization
of the neural network model for solving Backward Stochastic Differential Equations (BSDEs).
"""

import logging
import time
import json

import numpy as np
import torch
import torch.optim as optim

from . import equation as eqn
from .model import NonsharedModel

# Constant for clipping the loss delta to avoid numerical instability
DELTA_CLIP = 50.0


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
        self.eqn_config = config['eqn_config']
        self.net_config = config['net_config']
        self.bsde = bsde
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32

        # Initialize model and get reference to y_init parameter
        self.model = NonsharedModel(config, bsde, device=self.device, dtype=self.dtype)
        self.y_init = self.model.y_init

        # Setup optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.net_config['lr_values'][0], eps=1e-8)
        if len(self.net_config['lr_values']) > 1:
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.net_config['lr_boundaries'], gamma=1.0
            )
        else:
            self.lr_scheduler = None

    def train(self):
        """Trains the model using stochastic gradient descent.
        
        Performs training iterations, periodically evaluating on validation data
        and logging the results.

        Returns:
            numpy.ndarray: Training history containing step, loss, Y0 and elapsed time
        """
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config['valid_size'])
        valid_dw, valid_x = valid_data
        valid_dw = torch.tensor(valid_dw, dtype=self.dtype, device=self.device)
        valid_x = torch.tensor(valid_x, dtype=self.dtype, device=self.device)

        # Set model to evaluation mode for validation
        # self.model.eval()
        with torch.no_grad():
            loss = self.loss_fn(valid_dw, valid_x, training=False).item()
            # y_init = self.y_init.data.cpu().numpy()[0]
            y_init = self.y_init(valid_x[:, :, 0], training=False)
            y_init = y_init.data.cpu().numpy().mean()
            elapsed_time = time.time() - start_time
            training_history.append([0, loss, y_init, elapsed_time])
            if self.net_config['verbose']:
                # logging.info(f"step: {0:5},    loss: {loss:.4e}, Y0: {y_init:.4e},   elapsed time: {int(elapsed_time)}")
                print(f"step: {0:5},    loss: {loss:.6f}, Y0: {y_init:.6f},   elapsed time: {int(elapsed_time)}")

        # Set model back to training mode
        # self.model.train()

        # Main training loop
        for step in range(1, self.net_config['num_iterations'] + 1):
            # Sample training data
            train_data = self.bsde.sample(self.net_config['batch_size'])
            dw, x = train_data
            dw = torch.tensor(dw, dtype=self.dtype, device=self.device)
            x = torch.tensor(x, dtype=self.dtype, device=self.device)

            # Perform a training step
            self.train_step(dw, x)

            # Update learning rate scheduler if applicable
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Log progress at specified frequency
            if step % self.net_config['logging_frequency'] == 0:
                # self.model.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(valid_dw, valid_x, training=False).item()
                    # y_init = self.y_init.data.cpu().numpy()[0]
                    y_init = self.y_init(valid_x[:, :, 0], training=False)
                    y_init = y_init.data.cpu().numpy().mean()
                    elapsed_time = time.time() - start_time
                    training_history.append([step, val_loss, y_init, elapsed_time])
                    if self.net_config['verbose']:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        # logging.info(f"step: {step:5},    loss: {val_loss:.4e}, Y0: {y_init:.4e},   elapsed time: {int(elapsed_time)},   lr: {current_lr:.4e}")
                        print(f"step: {step:5},    loss: {val_loss:.6f}, Y0: {y_init:.6f},   elapsed time: {int(elapsed_time)},   lr: {current_lr:.6f}")
                # self.model.train()

        return np.array(training_history)

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
        y_terminal = self.model((dw, x), training)
        g_terminal = self.bsde.g_torch(torch.tensor(self.bsde.total_time, dtype=self.dtype, device=self.device), x[:, :, -1])
        delta = y_terminal - g_terminal
        abs_delta = torch.abs(delta)
        mask = abs_delta < DELTA_CLIP
        loss = torch.mean(torch.where(mask, delta ** 2, 2 * DELTA_CLIP * abs_delta - DELTA_CLIP ** 2))
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

    # -------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------
    # Initialize BSDE equation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    bsde = getattr(eqn, config['eqn_config']['eqn_name'])(config['eqn_config'], device=device, dtype=dtype)

    # Initialize and train BSDE solver
    solver = BSDESolver(config, bsde, device=device, dtype=dtype)
    
    # -------------------------------------------------------
    # Testing Model Components
    # -------------------------------------------------------
    # Test model initialization
    print("\nTesting model initialization...")
    print("Model architecture:", solver.model)
    print("Number of trainable variables:", len(list(solver.model.parameters())))
    print("Initial y_init value:", solver.y_init.data.cpu().numpy()[0])
    
    # Test data generation
    print("\nTesting data generation...")
    test_batch = bsde.sample(config['net_config']['batch_size'])
    test_dw, test_x = test_batch
    test_dw = torch.tensor(test_dw, dtype=solver.dtype, device=solver.device)
    test_x = torch.tensor(test_x, dtype=solver.dtype, device=solver.device)
    print("Sample batch shapes - dw:", test_dw.shape, "x:", test_x.shape)
    
    # Test forward pass
    print("\nTesting forward pass...")
    y_pred = solver.model((test_dw, test_x), training=False)
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
    print("Columns: [step, loss, Y0, elapsed_time]")
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
