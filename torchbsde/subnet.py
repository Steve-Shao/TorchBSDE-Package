import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForwardSubNet(nn.Module):
    """A feed-forward neural network with batch normalization layers.
    
    Current Implementation:
    -----------------------
    This network:
      - Takes an input of dimension `dim` (from config.equation_config.dim).
      - Applies an initial Batch Normalization (BN).
      - Then applies a sequence of (Linear -> BN -> ReLU) layers for each hidden layer.
      - Finally, applies a last Linear layer followed by a final BN.
    
    The final output dimension matches `dim` if is_derivative=True, else 1.

    Note on Implementation Details:
    -------------------------------
    While we do not change the current code logic, here are various design considerations 
    and details one could tune or specify if needed:
    
    1. Parameter Initialization:
       - Linear (Dense) layers currently rely on default initializers. Could be changed if desired.
       - BatchNormalization layers:
         * In TensorFlow, beta/gamma initializers were specified. We replicate this here by manually
           initializing BN parameters after layer creation.

    2. Batch Normalization:
       - momentum and epsilon values are replicated as closely as possible.
       - `momentum=0.99` in TF means the running averages update factor is 0.99.
         In PyTorch, momentum means something slightly different (factor applied to update),
         so we use momentum=0.01 (which corresponds to a similar update rate to TF's momentum=0.99).

    3. Activation Functions:
       - Currently using ReLU.
       - Could consider other activations if needed.

    4. Network Structure:
       - Number of hidden layers and their sizes come from config.
       - Could modify structure if needed, but not done here.

    5. Data Types and Computation:
       - Currently using float32 by default in PyTorch.
       - Could specify device (CPU/GPU), dtype, etc.

    6. Other Techniques:
       - Similar considerations as the TF implementation, just PyTorch-based now.

    The logic and structure remain identical to the TensorFlow version, only the framework changed.
    """

    def __init__(self, config, device=None, dtype=None, is_derivative=True):
        super(FeedForwardSubNet, self).__init__()
        dim = config['equation_config']['dim']
        num_hiddens = config['network_config']['num_hiddens']
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32

        # ---------------------------------------------------------------------
        # Get BN usage flags from config, with default True if missing
        # ---------------------------------------------------------------------
        self.use_bn_input = config['network_config'].get('use_bn_input', True)
        self.use_bn_hidden = config['network_config'].get('use_bn_hidden', True)
        self.use_bn_output = config['network_config'].get('use_bn_output', True)
        self.use_shallow_y_init = config['network_config'].get('use_shallow_y_init', True)
        
        # ---------------------------------------------------------------------
        # Get careful NN initialization flag from config, default to False
        # ---------------------------------------------------------------------
        self.careful_nn_initialization = config['network_config'].get('careful_nn_initialization', False)

        # ---------------------------------------------------------------------
        # Set activation function with configurable parameters
        # ---------------------------------------------------------------------
        activation_function_str = config['network_config'].get('activation_function', 'ReLU')
        
        # Get activation function parameters from config (if provided)
        activation_params = config['network_config'].get('activation_params', {})
        
        # Create a dictionary of activation functions with their parameters
        activation_functions = {
            'ReLU': lambda x: F.relu(x),
            'LeakyReLU': lambda x: F.leaky_relu(x, negative_slope=activation_params.get('leakyrelu_negative_slope', 0.01)),
            'Sigmoid': lambda x: torch.sigmoid(x),
            'Tanh': lambda x: torch.tanh(x),
            'ELU': lambda x: F.elu(x, alpha=activation_params.get('elu_alpha', 1.0)),
            'GELU': lambda x: F.gelu(x),
            'SELU': lambda x: F.selu(x),
            'SiLU': lambda x: F.silu(x),  # Swish activation
            'Swish': lambda x: F.silu(x),  # Alternative name for SiLU
            'Softmax': lambda x: F.softmax(x, dim=activation_params.get('softmax_dim', -1)),
            'Hardswish': lambda x: F.hardswish(x),
            'Hardtanh': lambda x: F.hardtanh(x, 
                                min_val=activation_params.get('hardtanh_min_val', -1.0), 
                                max_val=activation_params.get('hardtanh_max_val', 1.0)),
            'Hardsigmoid': lambda x: F.hardsigmoid(x),
            # Add more activation functions here if needed
        }
        
        if activation_function_str not in activation_functions:
            raise ValueError(f"Invalid activation function: {activation_function_str}")
        self.activation = activation_functions[activation_function_str]

        out_features = dim if is_derivative else 1
        # If not derivative, we only use the first hidden layer
        num_hiddens = num_hiddens if is_derivative and self.use_shallow_y_init else [num_hiddens[0]]

        # ---------------------------------------------------------------------
        # Create Batch Normalization layers
        # ---------------------------------------------------------------------
        self.bn_layers = nn.ModuleList()
        
        # (1) Input BN
        if self.use_bn_input:
            self.bn_layers.append(
                nn.BatchNorm1d(dim, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype)
            )
        else:
            self.bn_layers.append(None)

        # (2) BN for each hidden layer
        for h in num_hiddens:
            if self.use_bn_hidden:
                self.bn_layers.append(
                    nn.BatchNorm1d(h, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype)
                )
            else:
                self.bn_layers.append(None)

        # (3) Output BN
        if self.use_bn_output:
            self.bn_layers.append(
                nn.BatchNorm1d(out_features, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype)
            )
        else:
            self.bn_layers.append(None)

        # ---------------------------------------------------------------------
        # Create linear (dense) layers
        # ---------------------------------------------------------------------
        self.dense_layers = nn.ModuleList()
        in_features = dim
        for h in num_hiddens:
            layer = nn.Linear(in_features, h, bias=False, device=self.device, dtype=self.dtype)
            self.dense_layers.append(layer)
            in_features = h
        self.dense_layers.append(
            nn.Linear(in_features, out_features, bias=False, device=self.device, dtype=self.dtype)
        )

        # ---------------------------------------------------------------------
        # Initialize BN parameters (if BN is used) to mimic TF initialization
        # gamma_initializer: uniform(0.1, 0.5), beta_initializer: normal(0.0, 0.1)
        # ---------------------------------------------------------------------
        for bn_layer in self.bn_layers:
            if bn_layer is not None:  # Skip if BN is disabled
                if bn_layer.weight is not None:
                    nn.init.uniform_(bn_layer.weight, 0.1, 0.5)  # gamma
                if bn_layer.bias is not None:
                    nn.init.normal_(bn_layer.bias, 0.0, 0.1)     # beta
                # moving_mean / moving_variance handled automatically by PyTorch

        # ---------------------------------------------------------------------
        # Apply careful initialization if enabled
        # ---------------------------------------------------------------------
        if self.careful_nn_initialization:
            # ReLU-like activations (use Kaiming/He initialization)
            if activation_function_str in ['ReLU', 'LeakyReLU', 'ELU', 'GELU', 'SELU', 'SiLU', 'Swish', 'Hardswish']:
                for layer in self.dense_layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), nonlinearity='relu')
            # Sigmoid/Tanh-like activations (use Xavier/Glorot initialization)
            elif activation_function_str in ['Sigmoid', 'Tanh', 'Hardsigmoid', 'Hardtanh', 'Softmax']:
                for layer in self.dense_layers:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
            else:
                raise ValueError(f"Unsupported activation function for initialization: {activation_function_str}")

    def forward(self, x, training):
        # Handle training/eval mode
        if training:
            self.train()
        else:
            self.eval()
        
        # ----------------------------------------------------------
        # 1) Input BN (self.bn_layers[0])
        # ----------------------------------------------------------
        if self.bn_layers[0] is not None:
            x = self.bn_layers[0](x)

        # ----------------------------------------------------------
        # 2) Hidden layers: Linear -> BN -> Activation
        # ----------------------------------------------------------
        # Because we have len(num_hiddens) hidden layers, they occupy
        # bn_layers indices 1..len(num_hiddens).
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            if self.bn_layers[i + 1] is not None:  # i+1 for BN index in hidden
                x = self.bn_layers[i + 1](x)
            
            # Apply activation function (now simplified since parameters are handled by lambda)
            x = self.activation(x)

        # ----------------------------------------------------------
        # 3) Final layer -> Output BN
        # ----------------------------------------------------------
        x = self.dense_layers[-1](x)
        if self.bn_layers[-1] is not None:
            x = self.bn_layers[-1](x)

        return x


if __name__ == "__main__":
    # -------------------------------------------------------
    # Prepare Mock Configuration
    # -------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    mock_config = {
        'equation_config': {'dim': 10},
        'network_config': {
            'num_hiddens': [20, 20],
            'use_bn_input': True,
            'use_bn_hidden': True,
            'use_bn_output': True,
            'use_shallow_y_init': True,
            'careful_nn_initialization': True,  # Set to True to enable careful initialization
            'activation_function': 'LeakyReLU',
            'activation_params': {'negative_slope': 0.2}  # Configure LeakyReLU's slope
        }
    }

    # Initialize the network
    net = FeedForwardSubNet(mock_config, device=device, dtype=dtype)
    # -------------------------------------------------------
    # Since we don't have a summary method as in TF, we can just print the model
    # structure by listing parameters and shapes.
    print("\nModel Structure:")
    print("---------------")
    print(net)
    print("---------------\n")

    # -------------------------------------------------------
    # Check Parameter Initialization
    # -------------------------------------------------------
    print("Parameter Initialization Distributions:")

    # Dense layers
    for idx, layer in enumerate(net.dense_layers):
        w = layer.weight.detach().cpu().numpy()
        print(f"Dense layer {idx} weights shape: {w.shape}")
        print(f"  mean: {w.mean():.4f}, std: {w.std():.4f}")

    # BN layers (gamma=weight, beta=bias, moving_mean=running_mean, moving_variance=running_var)
    for idx, bn_layer in enumerate(net.bn_layers):
        if bn_layer is not None:
            gamma = bn_layer.weight.detach().cpu().numpy()
            beta = bn_layer.bias.detach().cpu().numpy()
            moving_mean = bn_layer.running_mean.detach().cpu().numpy()
            moving_variance = bn_layer.running_var.detach().cpu().numpy()
            print(f"BN layer {idx} gamma shape: {gamma.shape}, mean: {gamma.mean():.4f}, std: {gamma.std():.4f}")
            print(f"BN layer {idx} beta shape: {beta.shape}, mean: {beta.mean():.4f}, std: {beta.std():.4f}")
            print(f"BN layer {idx} moving_mean shape: {moving_mean.shape}, mean: {moving_mean.mean():.4f}, std: {moving_mean.std():.4f}")
            print(f"BN layer {idx} moving_variance shape: {moving_variance.shape}, mean: {moving_variance.mean():.4f}, std: {moving_variance.std():.4f}")

    # -------------------------------------------------------
    # Test Input/Output Characteristics
    # -------------------------------------------------------
    batch_size = 32
    input_dim = mock_config['equation_config']['dim']
    test_input_data = np.tile(np.linspace(0, 1, input_dim), (batch_size, 1)).astype(np.float32)
    print("\nInput Data Characteristics:")
    print(f"Input shape: {test_input_data.shape}")
    print(f"Input mean: {test_input_data.mean():.4f}, Input std: {test_input_data.std():.4f}")
    test_input = torch.tensor(test_input_data, device=device, dtype=dtype)

    # -------------------------------------------------------
    # Forward Pass in Training Mode
    # -------------------------------------------------------
    output_training = net(test_input, training=True).detach().cpu().numpy()
    print("\nOutput (Training Mode) Characteristics:")
    print(f"Output shape: {output_training.shape}")
    print(f"Output mean: {output_training.mean():.4f}, std: {output_training.std():.4f}")

    # -------------------------------------------------------
    # Forward Pass in Inference Mode
    # -------------------------------------------------------
    output_inference = net(test_input, training=False).detach().cpu().numpy()
    print("\nOutput (Inference Mode) Characteristics:")
    print(f"Output shape: {output_inference.shape}")
    print(f"Output mean: {output_inference.mean():.4f}, std: {output_inference.std():.4f}")

    # -------------------------------------------------------
    # Consistency Checks over Multiple Training Passes
    # -------------------------------------------------------
    print("\nTesting Batch Normalization Behavior Over Multiple Training Passes:")
    for i in range(3):
        output = net(test_input, training=True).detach().cpu().numpy()
        print(f"Training pass {i+1}: mean={output.mean():.4f}, std={output.std():.4f}")

    # -------------------------------------------------------
    # Parameter Count and Distribution After Forward Pass
    # -------------------------------------------------------
    print("\nModel Parameter Summary:")
    total_params = 0
    for idx, layer in enumerate(net.dense_layers):
        params_count = sum(p.numel() for p in layer.parameters())
        total_params += params_count
        print(f"Dense layer {idx}: parameter count={params_count}")
    for idx, bn_layer in enumerate(net.bn_layers):
        if bn_layer is not None:
            params_count = sum(p.numel() for p in bn_layer.parameters())
            # BN also has running_mean/var which are buffers (not parameters)
            # but to keep it similar to TF code, we only count parameters (weight, bias)
            total_params += params_count
            print(f"BN layer {idx}: parameter count={params_count}")
    print(f"Total trainable parameters (including BN): {total_params}")

    # -------------------------------------------------------
    # Test with Different Batch Sizes
    # -------------------------------------------------------
    print("\nTesting Different Batch Sizes:")
    for bsz in [1, 16, 64]:
        test_input_var = np.tile(np.linspace(0, 1, input_dim), (bsz, 1)).astype(np.float32)
        test_input_var = torch.tensor(test_input_var, device=device, dtype=dtype)
        out_var = net(test_input_var, training=False).detach().cpu().numpy()
        print(f"Batch size {bsz}: Input {test_input_var.shape} -> Output {out_var.shape}")
