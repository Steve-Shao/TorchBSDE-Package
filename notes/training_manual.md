# A Systematic Way to Improve Neural Network Training in DeepBSDE

### Step 0: Start with Minimal Representative Examples

### Step 1: Explore Hyperparameters

Quick trial and error and identify useful configurations and tricks. 

- **Objective**: Make training loss small and stable with fast convergence.
- **Controls**: Neural network architecture. Follow this order:
    1. Number of layers
    2. Number of nodes in each layer
    3. Activation function
- **Note**: Start with standard learning rate schedule and reference policy.
- **Potential issues and fixes**: If loss does not converge stably after tuning hyperparameters, consider these fixes:
    1. *Gradient clipping* - fixes exploding gradients (e.g. spikes in loss curve)
    2. *Batch normalization* - fixes covariate shift and slow convergence
    3. *Delta clipping in loss function* - handles extreme errors (outliers)
    4. ......

### Step 2: Improve Policy Performance

Find and fix issues.

- **Objective**: Improve trained neural network policy performance. 
- **Metric**: Consider these metrics for both limiting and stochastic systems:
    - Main metric: policy performance (revenue/reward/cost) vs benchmarks
    - Other metrics: values and visualizations of
        - State paths over time
        - Action paths over time
        - Gradients (on paths over time, or across states)
        - Initial value function
        - ......
- **Controls**: Reference policy
- **Note**: May need to return to Step 1 to tune hyperparameters for different reference policies.
- **Potential issues and fixes**: For persistent policy issues, consider these fixes:
    1. *Shape constraints* - penalizes invalid gradient subnet values
    2. *Smoothing of non-smooth functions* - fixes gradient subnet discontinuities
    3. ......

### Step 3: Exploitation

Push to the limit. 

- **Objective**: Maximize policy performance and minimize training loss.
- **Controls**: Learning rate schedule
- **Potential tricks**:
    1. Early stopping - prevents overfitting
    2. ......
