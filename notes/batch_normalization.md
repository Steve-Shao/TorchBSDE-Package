## Notes on Batch Normalization

### 1. What is Batch Normalization?

Batch Normalization (BN) is a technique to stabilize and accelerate the training of deep neural networks. It does so by normalizing the intermediate activations (inputs to subsequent layers) of the network. By maintaining stable distributions of layer inputs, BN helps mitigate issues such as the "internal covariate shift," often leading to faster convergence and better generalization.

**Mathematically:**

Given an input mini-batch $X = \{x_1, x_2, \ldots, x_m\}$ for a particular layer, where each $x_i$ is a vector (e.g., features of a single sample), batch normalization is computed as follows:

1. Compute the mean of each feature over the batch:
   $$
   \mu = \frac{1}{m} \sum_{i=1}^{m} x_i
   $$

2. Compute the variance of each feature over the batch:
   $$
   \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2
   $$

3. Normalize each feature:
   $$
   \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$
   Here, $\epsilon$ is a small positive constant to prevent division by zero.

4. Finally, a learnable linear transformation is applied to the normalized values:
   $$
   y_i = \gamma \hat{x}_i + \beta
   $$
   where $\gamma$ (scale) and $\beta$ (shift) are trainable parameters.

During training, $\mu$ and $\sigma^2$ are computed from the current mini-batch. During inference, fixed running averages of $\mu$ and $\sigma^2$ (collected during training) are used so that normalization does not depend on the batch being processed.

**References:**
- For PyTorch's BatchNorm1d implementation, see the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- For TensorFlow's BatchNormalization implementation, see the [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)


### 2. Differences between TensorFlow and PyTorch Implementations

While the concept and mathematics of batch normalization remain the same, some implementation details differ between TensorFlow (TF) and PyTorch (PT):

**Parameter Naming and Access:**
- **TensorFlow**:  
  - BatchNormalization layers maintain `gamma` (scale), `beta` (offset), as well as `moving_mean` and `moving_variance` as variables.  
  - Calling `layer.get_weights()` returns `[gamma, beta, moving_mean, moving_variance]`.
  - `moving_mean` and `moving_variance` are **not** trainable but are stored as variables for convenience and updated via exponential moving averages.
  
- **PyTorch**:  
  - `BatchNorm1d`, `BatchNorm2d`, etc., store `weight` and `bias` as trainable parameters (corresponding to `gamma` and `beta`).  
  - `running_mean` and `running_var` are maintained as *buffers*, not parameters. They are not returned by `layer.parameters()` and are not directly trainable. They are updated in-place during training.

**Momentum Interpretation:**
- **TensorFlow**:  
  - `momentum` parameter in `BatchNormalization` is the smoothing factor for updating running statistics. A typical value like `momentum=0.99` means the running averages are updated with a factor of `0.99` each step.
  
- **PyTorch**:  
  - `momentum` in `BatchNorm` is interpreted differently. A PyTorch momentum setting of `0.1` corresponds roughly to a TensorFlow momentum of `0.9`, and `0.01` in PyTorch aligns with `0.99` in TensorFlow. Adjusting these values when porting models between frameworks is often necessary to produce similar behaviors.

**Initialization:**
- **TensorFlow**:  
  - Allows specifying initializers for `gamma` and `beta` directly in `BatchNormalization` constructor (e.g., `gamma_initializer`, `beta_initializer`).
  
- **PyTorch**:  
  - Typically uses default initializers. For custom initialization, one must manually initialize `bn.weight` and `bn.bias` after creating the layer.

**When Implementing or Migrating Between Frameworks:**
- Ensure that the `momentum` parameter is adapted correctly to maintain similar running statistics behavior.
- Remember that PyTorch’s `running_mean` and `running_var` are not parameters; they do not appear in `list(model.parameters())`.
- If you rely on custom initializers in TensorFlow, you must replicate these initializations manually in PyTorch.
- The differences in parameter and buffer handling mean that the parameter count reported by each framework can differ, even though the conceptual BN layer is the same.



### 3. Additional Notes

**How Gamma ($\gamma$) and Beta ($\beta$) Are Learned and Updated**

The parameters $\gamma$ (scale) and $\beta$ (shift) in batch normalization allow the model to "un-normalize" the normalized outputs if it is beneficial for learning. Initially, after normalization, each feature in the batch has zero mean and unit variance, but $\gamma$ and $\beta$ give the network the flexibility to scale and shift these normalized features. Over training, these parameters are adjusted via gradient-based optimization (e.g., SGD, Adam) to improve model performance.

Algorithmic Steps (During Training):

1. **Forward Pass**:
   Given a mini-batch $X = \{x_1, \dots, x_m\}$, compute:
   $$
   \mu = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad
   \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu)^2
   $$
   Then normalize and shift:
   $$
   \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad
   y_i = \gamma \hat{x}_i + \beta
   $$

2. **Compute Loss**:
   The network outputs (including the BN-transformed features) are used to compute a loss function $\mathcal{L}$, for example cross-entropy loss for classification.

3. **Backward Pass (Gradients Computation)**:
   - Compute the gradient of the loss $\mathcal{L}$ with respect to the BN outputs $y_i$:
     $$
     \frac{\partial \mathcal{L}}{\partial y_i} \text{ is computed by backpropagation through the layers following BN.}
     $$

   - Since $y_i = \gamma \hat{x}_i + \beta$, the partial derivatives with respect to $\gamma$ and $\beta$ are:
     $$
     \frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i
     $$
     $$
     \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i}
     $$

   Here, we sum over the batch dimension since both $\gamma$ and $\beta$ are scalar parameters per feature dimension.

4. **Update Parameters**:
   Given a chosen optimizer (e.g., SGD with learning rate $\eta$), update each parameter:
   $$
   \gamma \leftarrow \gamma - \eta \frac{\partial \mathcal{L}}{\partial \gamma}, \quad
   \beta \leftarrow \beta - \eta \frac{\partial \mathcal{L}}{\partial \beta}
   $$

   In practice, more sophisticated optimizers like Adam might be used, which adjust $\eta$ adaptively:
   $$
   \gamma \leftarrow \gamma - \eta_\gamma(\text{grad}_\gamma), \quad
   \beta \leftarrow \beta - \eta_\beta(\text{grad}_\beta)
   $$

   Here $\text{grad}_\gamma$ and $\text{grad}_\beta$ represent the gradients computed above, and $\eta_\gamma, \eta_\beta$ represent effective learning rates for these parameters (potentially modified by adaptive methods).

5. **Summary**:
    - Forward pass: Normalize inputs, then scale and shift by $\gamma, \beta$.
    - Backward pass: Compute gradients $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$.
    - Parameter update: Use standard optimizer steps to update $\gamma$ and $\beta$.

This process repeats each training iteration, allowing $\gamma$ and $\beta$ to evolve and learn the best scaling and shifting to improve the network’s performance.  