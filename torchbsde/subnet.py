import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardSubNet(nn.Module):
    """A feed-forward neural network with batch normalization layers.
    
    Current Implementation:
    -----------------------
    This network:
      - Takes an input of dimension `dim` (from config.eqn_config.dim).
      - Applies an initial Batch Normalization (BN).
      - Then applies a sequence of (Linear -> BN -> ReLU) layers for each hidden layer.
      - Finally, applies a last Linear layer followed by a final BN.
    
    The final output dimension matches `dim`.

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

    def __init__(self, config, device=None, dtype=None):
        super(FeedForwardSubNet, self).__init__()
        dim = config['eqn_config']['dim']
        num_hiddens = config['net_config']['num_hiddens']
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32

        # Create batch normalization layers:
        # In TF: len(num_hiddens) + 2 BN layers
        # For PyTorch BatchNorm1d, we must specify the number of features for each BN layer.
        # Initial BN: input dimension is `dim`
        # Then each hidden BN: dimension matches the corresponding hidden layer
        # Final BN: dimension = `dim`
        self.bn_layers = nn.ModuleList()
        self.bn_layers.append(nn.BatchNorm1d(dim, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype))
        for h in num_hiddens:
            self.bn_layers.append(nn.BatchNorm1d(h, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype))
        self.bn_layers.append(nn.BatchNorm1d(dim, eps=1e-6, momentum=0.01, device=self.device, dtype=self.dtype))

        # Create linear (dense) layers:
        # First hidden layer: from dim -> num_hiddens[0]
        # Subsequent hidden layers: from num_hiddens[i-1] -> num_hiddens[i]
        # Final layer: from num_hiddens[-1] -> dim
        self.dense_layers = nn.ModuleList()
        in_features = dim
        for h in num_hiddens:
            layer = nn.Linear(in_features, h, bias=False, device=self.device, dtype=self.dtype)
            self.dense_layers.append(layer)
            in_features = h
        self.dense_layers.append(nn.Linear(in_features, dim, bias=False, device=self.device, dtype=self.dtype))

        # Initialize BN parameters to mimic TF initialization:
        # gamma_initializer: uniform(0.1, 0.5)
        # beta_initializer: normal(0.0, 0.1)
        for bn_layer in self.bn_layers:
            if bn_layer.weight is not None:
                nn.init.uniform_(bn_layer.weight, 0.1, 0.5)  # gamma
            if bn_layer.bias is not None:
                nn.init.normal_(bn_layer.bias, 0.0, 0.1)     # beta
            # moving_mean and moving_variance are handled automatically by PyTorch

    def forward(self, x, training):
        # In PyTorch, to set training/inference mode for BN, we use net.train() or net.eval().
        # Here, we rely on the 'training' argument:
        # If training=True, we call self.train(), else self.eval(), before the forward pass outside.
        # Since the user wants minimal changes and identical logic, we handle mode externally.
        
        # Initial BN
        x = self.bn_layers[0](x)

        # Hidden layers: Linear -> BN -> ReLU
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x)
            x = F.relu(x)

        # Final layer and BN
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x)
        return x


if __name__ == "__main__":
    # -------------------------------------------------------
    # Prepare Mock Configuration
    # -------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    mock_config = {
        'eqn_config': {'dim': 10},
        'net_config': {'num_hiddens': [20, 20]}
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
    input_dim = mock_config['eqn_config']['dim']
    test_input_data = np.tile(np.linspace(0, 1, input_dim), (batch_size, 1)).astype(np.float32)
    print("\nInput Data Characteristics:")
    print(f"Input shape: {test_input_data.shape}")
    print(f"Input mean: {test_input_data.mean():.4f}, Input std: {test_input_data.std():.4f}")
    test_input = torch.tensor(test_input_data, device=device, dtype=dtype)

    # -------------------------------------------------------
    # Forward Pass in Training Mode
    # -------------------------------------------------------
    net.train()  # Set model to training mode for BN
    output_training = net(test_input, training=True).detach().cpu().numpy()
    print("\nOutput (Training Mode) Characteristics:")
    print(f"Output shape: {output_training.shape}")
    print(f"Output mean: {output_training.mean():.4f}, std: {output_training.std():.4f}")

    # -------------------------------------------------------
    # Forward Pass in Inference Mode
    # -------------------------------------------------------
    net.eval()  # Set model to inference mode for BN
    output_inference = net(test_input, training=False).detach().cpu().numpy()
    print("\nOutput (Inference Mode) Characteristics:")
    print(f"Output shape: {output_inference.shape}")
    print(f"Output mean: {output_inference.mean():.4f}, std: {output_inference.std():.4f}")

    # -------------------------------------------------------
    # Consistency Checks over Multiple Training Passes
    # -------------------------------------------------------
    print("\nTesting Batch Normalization Behavior Over Multiple Training Passes:")
    net.train()
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
    net.eval()
    for bsz in [1, 16, 64]:
        test_input_var = np.tile(np.linspace(0, 1, input_dim), (bsz, 1)).astype(np.float32)
        test_input_var = torch.tensor(test_input_var, device=device, dtype=dtype)
        out_var = net(test_input_var, training=False).detach().cpu().numpy()
        print(f"Batch size {bsz}: Input {test_input_var.shape} -> Output {out_var.shape}")
