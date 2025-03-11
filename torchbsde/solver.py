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
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from . import equation as eqn
from .model import NonsharedModel


class BSDESolver:
    """
    Solver class that trains neural networks to solve BSDEs.

    This class orchestrates the training process, including:
        - Checking/creating an experiment directory
        - Loading a previously saved experiment (if it exists)
        - Initializing model, optimizer, and scheduler
        - Executing training steps and tracking training history
        - Saving states (model, optimizer, scheduler, etc.) for resumability

    Args:
        config (dict): Configuration dictionary containing model and equation parameters
        bsde (object): BSDE equation object defining the problem to solve
        device (torch.device, optional): Device to run computations on (CPU or GPU).
                                         Defaults to CUDA if available, else CPU.
        dtype (torch.dtype, optional): Data type for PyTorch tensors.
                                       Defaults to float32 if not specified.
    """
    def __init__(self, config, bsde, device=None, dtype=None):
        # -----------------------------------------
        # 1. Basic Attributes
        # -----------------------------------------
        self.config = config
        self.bsde = bsde
        self.dtype = dtype or torch.float32
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timezone = pytz.timezone(config.get('timezone', 'UTC'))
        
        # Paths for experiment directory
        self.test_folder_path = self.config.get('test_folder_path', 'tests/')
        self.test_scenario_name = self.config.get('test_scenario_name', 'new_test')
        self.exp_dir = None  # Will be set in _create_experiment_directory or loaded

        self.lr_plateau_warmup_step = self.config['solver_config'].get('lr_plateau_warmup_step', 0)
        self.lr_decay_rate = self.config['solver_config'].get('lr_decay_rate', 0.5)
        self.lr_plateau_patience = self.config['solver_config'].get('lr_plateau_patience', 10)
        self.lr_plateau_threshold = self.config['solver_config'].get('lr_plateau_threshold', 1e-4)
        self.lr_plateau_cooldown = self.config['solver_config'].get('lr_plateau_cooldown', 10)
        self.lr_plateau_min_lr = self.config['solver_config'].get('lr_plateau_min_lr', 1e-5)
        
        # Extract checkpoint frequency from config (default: 1000)
        self.checkpoint_frequency = self.config['solver_config'].get('checkpoint_frequency', 1000)
        
        # Get validation seed (default: 42)
        self.valid_seed = self.config['solver_config'].get('valid_seed', 42)

        # -----------------------------------------
        # 2. Detecting and Handling Existing Experiments
        # -----------------------------------------
        # No existing experiment: setup fresh
        self._create_experiment_directory()
        self._initialize_logger()
        self._extract_subconfigs()
        self._initialize_model()
        self._initialize_optimizer_and_scheduler()

        self.training_start_step = 0 

        # Existing experiment found: validate config and load everything
        if self._experiment_exists():
            self._validate_config()
            self._load_existing_experiment()

        self._write_config_to_disk()

    # =====================================================================
    #                       PRIVATE HELPER METHODS
    # =====================================================================

    def _create_experiment_directory(self):
        """
        Creates the experiment directory if it doesn't exist. 
        This directory will store config, logs, model checkpoints, etc.
        """
        self.exp_dir = os.path.join(self.test_folder_path, self.test_scenario_name)
        os.makedirs(self.exp_dir, exist_ok=True)

    def _initialize_logger(self):
        """
        Sets up the logger for the solver, including file and console handlers.
        Logs basic initialization info (e.g., directory, date/time, device, dtype).
        """
        self.logger = logging.getLogger(self.test_scenario_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplication
        self.logger.handlers = []
        self.logger.propagate = False

        log_file = os.path.join(self.exp_dir, 'log.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("========== Experiment Information ==========")
        self.logger.info(f"Experiment directory : {self.exp_dir}")
        self.logger.info(f"Date                 : {datetime.now(self.timezone).strftime('%Y-%m-%d')}")
        self.logger.info(f"Time                 : {datetime.now(self.timezone).strftime('%H:%M:%S')}")
        self.logger.info(f"Dtype                : {self.dtype}")
        self.logger.info(f"Device               : {self.device}")

    def _extract_subconfigs(self):
        """
        Extracts relevant subconfigs for solver usage.
        """

        # Extract sub-configurations
        self.equation_config = self.config['equation_config']
        self.network_config = self.config['network_config']
        self.solver_config = self.config['solver_config']

        # Solver attributes
        self.negative_grad_penalty = self.solver_config.get('negative_grad_penalty', 0.0)
        self.delta_clip = self.solver_config.get('delta_clip', 50.0)

    def _initialize_model(self):
        """
        Initializes the neural network model from NonsharedModel.
        """
        self.model = NonsharedModel(
            self.config,
            self.bsde,
            device=self.device,
            dtype=self.dtype
        )
        # For convenience, store a reference to the y_init sub-network
        self.y_init = self.model.y_init

    def _initialize_optimizer_and_scheduler(self):
        """
        Initializes the optimizer (Adam) and the optional learning rate scheduler.
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.solver_config['lr_start_value'],
            eps=1e-8
        )

        # Get scheduler type from config
        self.lr_scheduler_type = self.config['solver_config'].get("lr_scheduler", "manual")
        
        # Initialize LR scheduler based on the scheduler type
        if self.lr_scheduler_type == "manual":
            lr_boundaries = self.solver_config.get('lr_boundaries', [])
            if lr_boundaries:
                decay_rate = self.solver_config['lr_decay_rate']
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=lr_boundaries,
                    gamma=decay_rate
                )
                self.logger.info(
                    f"Manual LR scheduler initialized with milestones {lr_boundaries} "
                    f"and gamma {decay_rate}"
                )
            else:
                self.lr_scheduler = None
                self.logger.info("Manual LR scheduler not initialized due to empty lr_boundaries.")
        elif self.lr_scheduler_type == "reduce_on_plateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode = 'min',
                factor = self.lr_decay_rate,
                patience = self.lr_plateau_patience, 
                threshold = self.lr_plateau_threshold,
                threshold_mode = 'rel',
                cooldown = self.lr_plateau_cooldown, 
                min_lr = self.lr_plateau_min_lr
            )
            self.logger.info("ReduceLROnPlateau scheduler initialized.")
        else:
            raise ValueError(f"Unsupported lr_scheduler type: {self.lr_scheduler_type}")

    # =====================================================================
    #                 LOADING EXISTING EXPERIMENT METHODS
    # =====================================================================

    def _experiment_exists(self):
        """
        Checks if an experiment directory and its required files already exist.
        
        Required files: 'config.json', 'model.pth', 'optimizer.pth', 'training_history.csv'
        
        Returns:
            bool: True if all required files exist, False otherwise.
        """
        self.exp_dir = os.path.join(self.test_folder_path, self.test_scenario_name)
        required_files = ['config.json', 'model.pth', 'optimizer.pth', 'training_history.csv']
        return all(os.path.exists(os.path.join(self.exp_dir, file)) for file in required_files)

    def _validate_config(self):
        """
        Validates that the user-provided config matches critical fields of the saved config.
        
        This check ensures consistency in equation_config, network_config, dtype, device, etc.
        For solver_config, only num_iterations or lr_boundaries are allowed to differ (as
        they can be extended).
        
        Raises:
            ValueError: If there's a mismatch in critical config fields.
        """
        self.logger.info("========== Existing Checkpoint Found ==========")
        self.logger.info("Validating previous state...")

        # Load existing config
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        input_config = self.config.copy()
        saved_solver_config = saved_config.get('solver_config', {})
        input_solver_config = input_config.get('solver_config', {})

        # 1. Check solver_config keys (except 'num_iterations' and 'lr_boundaries')
        for key in saved_solver_config:
            if key not in ['num_iterations', 'lr_boundaries']:
                if key in input_solver_config:
                    # If both have the same key, check if they differ
                    if saved_solver_config[key] != input_solver_config[key]:
                        raise ValueError(
                            f"Mismatch in solver_config key '{key}': "
                            f"saved value {saved_solver_config[key]}, "
                            f"input value {input_solver_config[key]}"
                        )

        # # 2. Check equation_config
        # if saved_config.get('equation_config') != input_config.get('equation_config'):
        #     raise ValueError("Mismatch in equation_config between saved config and input config.")

        # 3. Check network_config
        if saved_config.get('network_config') != input_config.get('network_config'):
            raise ValueError("Mismatch in network_config between saved config and input config.")

        # 4. Check dtype
        if saved_config.get('dtype') != input_config.get('dtype'):
            raise ValueError(
                f"Mismatch in dtype: saved {saved_config.get('dtype')}, "
                f"input {input_config.get('dtype')}"
            )

        # 5. Check device
        saved_device = saved_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        input_device = str(self.device)
        if saved_device != input_device:
            raise ValueError(
                f"Mismatch in device: saved {saved_device}, input {input_device}"
            )

        # 6. Check timezone
        if saved_config.get('timezone') != input_config.get('timezone'):
            raise ValueError("Mismatch in timezone between saved config and input config.")

    def _load_existing_experiment(self):
        """
        Loads an existing experiment from disk, including:
            - config.json
            - model state (model.pth)
            - optimizer state (optimizer.pth)
            - LR scheduler state (scheduler.pth), if present
            - training_history.csv

        Updates internal attributes (config, device, dtype) and resumes
        training from the last recorded step (training_start_step).
        """
        self.logger.info("Loading previous state...")

        # Load existing config
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        # Load model weights
        model_path = os.path.join(self.exp_dir, 'model.pth')
        model_state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_state)
        self.logger.info(f"Model loaded from {model_path}")

        # Load optimizer state
        optimizer_path = os.path.join(self.exp_dir, 'optimizer.pth')
        optimizer_state = torch.load(optimizer_path, map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(optimizer_state)
        self.logger.info(f"Optimizer loaded from {optimizer_path}")

        # Load training history and compute next start step
        history_path = os.path.join(self.exp_dir, 'training_history.csv')
        self.training_history = np.loadtxt(history_path, delimiter=",", skiprows=1)
        self.training_start_step = int(self.training_history[-1, 0]) + 1
        self.logger.info(f"Training history loaded. Resuming from step {self.training_start_step}.")

        # Update number of iterations if provided
        if 'num_iterations' in self.solver_config:
            new_num_iterations = self.solver_config['num_iterations']
            saved_num_iterations = saved_config['solver_config'].get('num_iterations', 0)
            # If the user wants to train more steps, update the saved config
            if new_num_iterations < saved_num_iterations:
                raise ValueError("New num_iterations must be greater than the saved num_iterations.")

        # Load scheduler state if it exists and scheduler is initialized
        scheduler_path = os.path.join(self.exp_dir, 'scheduler.pth')
        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            scheduler_state = torch.load(scheduler_path, map_location=self.device, weights_only=True)
            self.lr_scheduler.load_state_dict(scheduler_state)
            self.logger.info(f"LR scheduler loaded from {scheduler_path}")
            
            if self.lr_scheduler_type == "manual": 
                # Process new learning rate boundaries if any exist
                new_boundaries = self.solver_config.get('lr_boundaries', [])
                if new_boundaries:
                    current_milestones = self.lr_scheduler.milestones
                    
                    # Validate existing milestones
                    old_boundaries = [b for b in new_boundaries if b <= self.training_start_step]
                    old_milestones = {k: v for k, v in current_milestones.items() 
                                    if k <= self.training_start_step}
                    
                    if set(old_boundaries) != set(old_milestones.keys()):
                        raise ValueError("Mismatch between old milestones in saved scheduler and current config")
                    
                    # Add any new milestone boundaries
                    new_steps = [b for b in new_boundaries if b > self.training_start_step and b not in current_milestones]
                    if new_steps:
                        for step in new_steps:
                            self.lr_scheduler.milestones[step] = self.lr_scheduler.milestones.get(step, 0) + 1
                        self.logger.info(f"Added new milestones: {new_steps}")
                
        elif self.lr_scheduler is not None:
            self.logger.info("LR scheduler state file not found. Using freshly initialized scheduler.")
            self.lr_scheduler.last_epoch = self.training_start_step - 1
        else:
            self.logger.info("No LR scheduler in use.")

    def _write_config_to_disk(self):
        """
        Saves the current config dictionary to config.json in the experiment directory.
        Also extracts relevant subconfigs for solver usage.
        """
        # Convert numpy arrays to lists for JSON serialization
        config_to_save = self.config.copy()
        for key, value in config_to_save.get('equation_config', {}).items():
            if isinstance(value, np.ndarray):
                config_to_save['equation_config'][key] = value.tolist()

        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")


    # =====================================================================
    #                       TRAINING AND LOSS
    # =====================================================================

    def train(self):
        """
        Orchestrates the training loop for the solver.

        The training loop proceeds from `self.training_start_step` up to 
        `self.solver_config['num_iterations']`. On each step:
            - Samples a training batch
            - Performs a forward pass and computes the loss
            - Applies gradients using the optimizer
            - Updates the LR scheduler if applicable
            - Logs and records training/validation loss at specified intervals
        
        Returns:
            numpy.ndarray: The full training history array with shape 
                           [num_records, 6], where columns are:
                           [step, train_loss, val_loss, y0, learning_rate, elapsed_time].
        """
        self.logger.info("========== Training Started ==========")
        start_time = time.time()
        
        # If we have existing history, convert to list for easy append, else start fresh
        training_history = self.training_history.tolist() if hasattr(self, 'training_history') else []

        # Prepare validation data with fixed seed
        valid_size = self.solver_config['valid_size']
        
        # Save current random states
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        if torch.cuda.is_available():
            torch_cuda_rng_state = torch.cuda.get_rng_state()
            
        # Set seed for validation data
        torch.manual_seed(self.valid_seed)
        np.random.seed(self.valid_seed)
        
        # Generate validation data with fixed seed
        valid_dw, valid_x, valid_u = self.bsde.sample(valid_size)
        valid_dw = valid_dw.to(dtype=self.dtype, device=self.device)
        valid_x = valid_x.to(dtype=self.dtype, device=self.device)
        if valid_u is not None:
            valid_u = valid_u.to(dtype=self.dtype, device=self.device)
            
        # Restore random states
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state)

        # Main training loop
        total_steps = self.solver_config['num_iterations']
        for step in range(self.training_start_step, total_steps + 1):
            # 1. Sample a training batch
            train_dw, train_x, train_u = self.bsde.sample(self.solver_config['batch_size']) 
            train_dw = train_dw.to(dtype=self.dtype, device=self.device)
            train_x = train_x.to(dtype=self.dtype, device=self.device)
            if train_u is not None:
                train_u = train_u.to(dtype=self.dtype, device=self.device)

            # Record train_loss for steps not logged as validation steps
            train_loss, train_squared_loss = self.loss_fn(train_dw, train_x, train_u, step, training=True)
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time

            # 2. Logging & validation at specified intervals
            if step % self.solver_config['logging_frequency'] == 0:
                with torch.no_grad():
                    # train_loss, train_squared_loss = self.loss_fn(train_dw, train_x, train_u, step, training=True)
                    # current_lr = self.optimizer.param_groups[0]['lr']
                    # elapsed_time = time.time() - start_time
                    val_loss, val_squared_loss = self.loss_fn(valid_dw, valid_x, valid_u, step, training=False)
                    y_init_val = self.y_init(valid_x[:, :, 0], training=False)
                    y_init = y_init_val.data.cpu().numpy().mean()
                    training_history.append([
                        step,
                        train_squared_loss,
                        val_squared_loss,
                        val_loss.item(), 
                        y_init,
                        current_lr,
                        elapsed_time
                    ])
                    if self.solver_config.get('verbose', False):
                        self.logger.info(
                            f"step: {step:5}, val loss: {val_squared_loss:.6f}, "
                            f"Y0: {y_init:.6f}, lr: {current_lr:.6f}"
                            # f"elapsed time: {int(elapsed_time)}"
                        )
                # 4. Step the LR scheduler based on the scheduler type
                if self.lr_scheduler:
                    if self.lr_scheduler_type == "reduce_on_plateau":
                        if step >= self.lr_plateau_warmup_step:
                            # self.lr_scheduler.step(val_loss)
                            self.lr_scheduler.step(val_squared_loss)
            else:
                # # Record train_loss for steps not logged as validation steps
                # train_loss, train_squared_loss = self.loss_fn(train_dw, train_x, train_u, step, training=True)
                # current_lr = self.optimizer.param_groups[0]['lr']
                # elapsed_time = time.time() - start_time
                training_history.append([
                    step,
                    train_squared_loss,
                    float('nan'),
                    float('nan'),
                    float('nan'),
                    current_lr,
                    elapsed_time
                ])

            # 3. Perform a single training step
            self._train_step(train_dw, train_x, train_u, step)

            # 4. Step the LR scheduler based on the scheduler type
            if self.lr_scheduler:
                if self.lr_scheduler_type == "manual":
                    self.lr_scheduler.step()
                    
            # 5. Save checkpoint at specified intervals
            if step % self.checkpoint_frequency == 0 and step > 0:
                self.logger.info(f"Saving checkpoint at step {step}")
                self.training_history = np.array(training_history)  # Update training_history before saving
                self.save_results()

        # Convert history to numpy and store it
        self.training_history = np.array(training_history)

    def _train_step(self, dw, x, u, step):
        """
        Performs one training step, consisting of:
            - Computing the loss
            - Zeroing gradients
            - Backpropagating
            - Stepping the optimizer
        
        Args:
            dw (torch.Tensor): Brownian increments (batch)
            x  (torch.Tensor): State process (batch)
        """
        loss, plain_loss = self.loss_fn(dw, x, u, step, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def loss_fn(self, dw, x, u, step, training):
        """
        Computes the loss for a given (dw, x) pair.
        
        The loss is primarily mean-squared error of the difference between
        the predicted terminal value and the true terminal value g_terminal.
        Large deviations are linearly approximated beyond 'self.delta_clip'.
        An additional penalty is applied to negative outputs if specified.

        Args:
            dw (torch.Tensor): Brownian increments
            x (torch.Tensor): State process
            training (bool): If True, applies training-time behaviors 
                             (e.g., dropout, batchnorm updates).

        Returns:
            torch.Tensor: The scalar loss.
            float: The plain squared loss without clipping or penalties.
        """
        # Forward pass: model returns (predicted terminal value, negative_grad_loss)
        y_terminal, negative_grad_loss = self.model((dw, x, u), step, training)

        # True terminal function g_terminal
        g_terminal = self.bsde.g_torch(
            torch.tensor(self.bsde.total_time, dtype=self.dtype, device=self.device),
            x[:, :, -1],
            step
        )

        # Compute delta = prediction - true_value
        delta = y_terminal - g_terminal
        abs_delta = torch.abs(delta)

        # Calculate plain squared loss
        plain_squared_loss = torch.mean(delta**2).item()

        # Mask for deltas within the clipping threshold
        mask = abs_delta < self.delta_clip

        # Quadratic region for small delta; linear region for large delta
        clipped_loss = torch.where(
            mask,
            delta**2,
            2.0 * self.delta_clip * abs_delta - (self.delta_clip**2)
        )

        # Mean over the batch
        loss = torch.mean(clipped_loss)

        # Add penalty for negative output if needed
        loss += self.negative_grad_penalty * negative_grad_loss
        return loss, plain_squared_loss

    # =====================================================================
    #                       SAVING AND PLOTTING
    # =====================================================================

    def save_results(self):
        """
        Saves all relevant training artifacts to disk, including:
            - training_history.csv
            - model.pth
            - optimizer.pth
            - scheduler.pth (if applicable)
            
        This should be called periodically or after training completes.
        """
        self.logger.info("========== Saving Results ==========")

        # Save training history
        if hasattr(self, 'training_history'):
            history_path = os.path.join(self.exp_dir, 'training_history.csv')
            np.savetxt(
                history_path,
                self.training_history,
                fmt=['%d', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e', '%.5e'],
                delimiter=",",
                header='step,train_squared_loss,val_squared_loss,val_loss,y0,learning_rate,elapsed_time',
                comments=''
            )
            self.logger.info(f"Training history saved to {history_path}")
        else:
            self.logger.error("No training history found. Please run training first.")

        # Save model state
        try:
            model_path = os.path.join(self.exp_dir, 'model.pth')
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Model state saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model state: {e}")

        # Save optimizer state
        try:
            optimizer_path = os.path.join(self.exp_dir, 'optimizer.pth')
            torch.save(self.optimizer.state_dict(), optimizer_path)
            self.logger.info(f"Optimizer state saved to {optimizer_path}")
        except Exception as e:
            self.logger.error(f"Failed to save optimizer state: {e}")

        # Save scheduler state based on the scheduler type
        if self.lr_scheduler:
            try:
                scheduler_path = os.path.join(self.exp_dir, 'scheduler.pth')
                torch.save(self.lr_scheduler.state_dict(), scheduler_path)
                self.logger.info(f"LR scheduler state saved to {scheduler_path}")
            except Exception as e:
                self.logger.error(f"Failed to save LR scheduler state: {e}")
        else:
            self.logger.warning("LR scheduler is not initialized; skipping save.")

    def plot_y0_history(self, filename='training_history.csv'):
        """
        Plots the Y0 (initial value) vs. steps.
        
        Args:
            filename (str): CSV filename in the experiment directory containing 
                            training history to plot.
        """
        csv_path = os.path.join(self.exp_dir, filename)
        if not os.path.exists(csv_path):
            self.logger.error("Training history CSV file not found. Please run training first.")
            return

        df = pd.read_csv(csv_path)

        # Get Y0 values, dropping NaN values
        y0_df = df[['step', 'y0']].dropna()

        # Plot setup
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y0_df['step'], y0_df['y0'], label='Y0')

        # Identify learning-rate change steps
        lr_changes = df[df['learning_rate'].diff() != 0]

        # Plot vertical lines at LR change steps
        y_min, y_max = ax.get_ylim()
        step_range = df['step'].max() - df['step'].min()
        offset = step_range * 0.01  # offset for text placement
        for _, row in lr_changes.iterrows():
            ax.axvline(x=row['step'], color='red', linestyle='--', alpha=0.7)
            ax.text(row['step'] + offset, y_max - offset,
                    f"LR: {row['learning_rate']:.3g}",
                    rotation=0, verticalalignment='top', color='red', fontsize=9)

        ax.set_xlabel('Step')
        ax.set_ylabel('Y0')
        ax.set_title('Initial Value (Y0) with Learning Rate Changes')
        ax.legend()
        plt.tight_layout()

        # Save and show plot
        plot_path = os.path.join(self.exp_dir, 'y0_history.png')
        plt.savefig(plot_path)
        plt.close()

        self.logger.info(f"Y0 history plot saved to {plot_path}")

    def plot_training_history(self, filename='training_history.csv', zoom_steps=200, zoom_alpha=0.1, zoom_scale='log', show_detail_plots=False):
        """Creates a combined plot showing full training history and zoomed segments around LR changes.
        
        Args:
            filename (str): CSV filename containing training history
            zoom_steps_window (int): Number of steps to show before/after each LR change in zoom plots
            zoom_alpha (float): Transparency for training loss in zoom plots
            zoom_scale (str): Scale for y-axis in zoom plots - 'log' or 'linear'
            show_detail_plots (bool): Whether to show detailed zoom plots in the second row
        """
        csv_path = os.path.join(self.exp_dir, filename)
        if not os.path.exists(csv_path):
            self.logger.error("Training history CSV file not found.")
            return

        # Load and prepare data
        df = pd.read_csv(csv_path)
        train_df = df[['step', 'train_squared_loss']].dropna()
        val_df = df[['step', 'val_squared_loss']].dropna()
        lr_changes = df[df['learning_rate'].diff() != 0]

        # Create figure with appropriate size and layout
        if show_detail_plots:
            fig = plt.figure(figsize=(15, 6))
            gs = plt.GridSpec(2, 1, height_ratios=[3, 3], figure=fig)
            ax_top = fig.add_subplot(gs[0, 0])
        else:
            fig = plt.figure(figsize=(12, 5))
            ax_top = fig.add_subplot(1, 1, 1)

        # Top plot: full history
        ax_top.plot(train_df['step'], train_df['train_squared_loss'], label='Training Loss')
        ax_top.plot(val_df['step'], val_df['val_squared_loss'], label='Validation Loss')
        ax_top.set_yscale('log')

        # Set y-axis limits and ticks
        min_loss = min(train_df['train_squared_loss'].min(), val_df['val_squared_loss'].min())
        max_loss = max(train_df['train_squared_loss'].max(), val_df['val_squared_loss'].max())
        y_min = 10**np.floor(np.log10(min_loss))
        y_max = 10**np.ceil(np.log10(max_loss))
        ax_top.set_ylim(y_min, y_max)
        ax_top.set_yticks(10**np.arange(np.floor(np.log10(y_min)), np.ceil(np.log10(y_max)) + 1))

        # Add learning rate change markers
        for _, row in lr_changes.iterrows():
            ax_top.axvline(x=row['step'], color='red', linestyle='--', alpha=0.7)
            ax_top.text(row['step'], y_max*0.99, f"LR: {row['learning_rate']:.3g}", 
                       rotation=270, va='top', color='red', fontsize=9)

        ax_top.set_xlabel('Step')
        ax_top.set_ylabel('Loss')
        ax_top.set_title('Training and Validation Loss with Learning Rate Changes')
        ax_top.legend()

        # Bottom plots: zoomed segments around LR changes (only if show_detail_plots is True)
        if show_detail_plots:
            # Skip the first LR change (initial LR)
            lr_changes = lr_changes.iloc[1:]
            num_changes = len(lr_changes)
            if num_changes > 0:
                gs_bottom = gs[1, 0].subgridspec(1, num_changes, wspace=0.05)  # Reduced horizontal spacing

                for i, (_, row) in enumerate(lr_changes.iterrows()):
                    ax = fig.add_subplot(gs_bottom[0, i])
                    
                    # Define window around LR change
                    center_step = row['step']
                    window_start = max(0, center_step - zoom_steps)
                    window_end = center_step + zoom_steps

                    # Plot segment data
                    mask = (train_df['step'] >= window_start) & (train_df['step'] <= window_end)
                    train_seg = train_df[mask]
                    val_seg = val_df[(val_df['step'] >= window_start) & (val_df['step'] <= window_end)]
                    
                    # Plot training loss with transparency
                    ax.plot(train_seg['step'], train_seg['train_squared_loss'], 
                           label='Training Loss', alpha=zoom_alpha)
                    ax.plot(val_seg['step'], val_seg['val_squared_loss'], 
                           label='Validation Loss')
                    ax.set_yscale(zoom_scale)

                    # Set y-axis limits based on validation loss only
                    val_min = val_seg['val_squared_loss'].min()
                    val_max = val_seg['val_squared_loss'].max()
                    
                    # Add margin based on scale type
                    if zoom_scale == 'log':
                        # For log scale, use multiplicative margin
                        margin_factor = 1.1  # 10% margin
                        ax.set_ylim(val_min / margin_factor, val_max * margin_factor)
                    else:
                        # For linear scale, use additive margin
                        margin = 0.1 * (val_max - val_min)  # 10% margin
                        ax.set_ylim(val_min - margin, val_max + margin)

                    # Add LR change marker
                    ax.axvline(x=center_step, color='red', linestyle='--', alpha=0.7)
                    
                    ax.set_xlabel('Step', fontsize=9)
                    ax.set_title(f'Cut for LR: {row["learning_rate"]:.3g}', fontsize=9)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.set_ylabel('')  # Remove y-axis label
                    ax.yaxis.set_visible(False)  # Hide entire y-axis

        plt.tight_layout()
        plot_path = os.path.join(self.exp_dir, 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Training history plot saved to {plot_path}")


# =====================================================================
#                  Example Usage (if run as main)
# =====================================================================
if __name__ == '__main__':
    # -----------------------------------------------------------------
    # Setup and Configuration
    # -----------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        "equation_config": {
            "_comment": "Example: HJBLQ equation",
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
            "valid_seed": 42,  # Added validation seed
            "lr_start_value": 1e-2,
            "lr_decay_rate": 0.5,
            "lr_boundaries": [1000, 2000, 3000],
            "num_iterations": 5000,
            "logging_frequency": 100,
            "checkpoint_frequency": 1000,  # Added checkpoint frequency
            "negative_grad_penalty": 0.0,
            "delta_clip": 50,
            "verbose": True,
        },
        "dtype": "float64",
        "test_folder_path": 'tests/',
        "test_scenario_name": 'new_test',
        "timezone": "America/Chicago",
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # Instantiate the equation
    bsde = getattr(eqn, config['equation_config']['eqn_name'])(
        config['equation_config'],
        device=device,
        dtype=dtype
    )

    # Create and train the solver
    solver = BSDESolver(config, bsde, device=device, dtype=dtype)

    # Test model initialization
    print("\nTesting model initialization...")
    print("Model architecture:", solver.model)
    print("Number of trainable parameters:", len(list(solver.model.parameters())))

    # Test data generation
    print("\nGenerating test batch...")
    test_batch = bsde.sample(config['solver_config']['batch_size'])
    if len(test_batch) == 3:
        test_dw, test_x, test_u = test_batch
    else:
        test_dw, test_x = test_batch
        test_u = None
    print("Sample batch shapes - dw:", test_dw.shape, "x:", test_x.shape)

    # Test forward pass
    print("\nTesting forward pass...")
    inputs = (test_dw, test_x, test_u) if test_u is not None else (test_dw, test_x, None)
    y_pred, negative_grad_loss = solver.model(inputs, step=0, training=False)
    print("Predicted Y-terminal shape:", y_pred.shape)

    # Test loss computation
    print("\nComputing loss...")
    loss_val, _ = solver.loss_fn(test_dw, test_x, test_u, step=0, training=False)
    print("Initial loss value:", loss_val.item())

    # Test full training
    print("\nStarting training...")
    training_history = solver.train()

    print("\nTraining completed.")
    print("Final training history shape:", training_history.shape)
    print("Columns: [step, train_loss, val_loss, y0, learning_rate, elapsed_time]")
    print("\nFinal metrics:")
    print("Last row in history:", training_history[-1])

    # Save final results
    solver.save_results()
    solver.plot_training_history()
