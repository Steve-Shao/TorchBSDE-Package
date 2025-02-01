# [DeepBSDE Solver](https://doi.org/10.1073/pnas.1718942115) Package in PyTorch

This repository implements Jiequn Han's [DeepBSDE solver](https://github.com/frankhan91/DeepBSDE) in `PyTorch`. It solves high-dimensional PDEs using deep learning. We have added the following improvements:

1. **Faster training on GPU**: All computations run on GPU with vectorized tensor operations. This speeds up training for large problems.
2. **Allow random initial state**: The initial state can be random, not just fixed. The initial value function uses a neural network instead of a scalar parameter.
3. **Allow "stop and resume" training**: Training can be paused and resumed by updating `num_iterations` in the config dict.
4. **Neural networks `z` represent gradient directly**: The networks output the gradient itself, not its product with the diffusion parameter `sigma`. This simplifies work with complex sigma equations.
5. **Removed engineering tricks**: We removed tricks like gradient input scaling from the original implementation.
6. **Allows reference policies**: This helps solve HJB equations for control problems.
7. **More config options**: The config dict now has these key sections:
  - `equation_config`: Basic equation settings:
    - `_comment`: Equation description
    - `eqn_name`: Equation name
    - `policy`: Reference policy for HJB control problems
    - ......
  - `network_config`: Neural network settings:
    - `use_bn_input`, `use_bn_hidden`, `use_bn_output`: Batch norm controls
    - `num_hiddens`: Hidden layer sizes
    - `activation_function`: Type of activation
    - ......
  - `solver_config`: Solver settings:
    - `batch_size`, `valid_size`: Training and validation batch sizes
    - `lr_scheduler`: Learning rate schedule (`manual` or `reduce_on_plateau`)
      - `lr_plateau_warmup_step`, `lr_plateau_patience`, `lr_plateau_threshold`, `lr_plateau_cooldown`, `lr_plateau_min_lr`: Plateau scheduler settings
      - `lr_start_value`, `lr_decay_rate`: Initial rate and decay
    - `num_iterations`: Total training steps
    - `logging_frequency`: Log interval
    - `negative_grad_penalty`: Penalty for negative gradients
    - ......
  - `dtype`: Data type for computations
  - `test_folder_path`: Test results folder
  - `test_scenario_name`: Test scenario ID
  - `timezone`: Logging timezone

The code has been restructured to work either as an installable Python package or as a git submodule in other projects. Extensive comments and docstrings have been added to enhance readability and understanding of the implementation. 

## Training Tips

We focus on solving the HJB equation for control problems, where a reference policy is used to generate training data. For those with a math background (like us), we suggest following these key principles in training:
1. Follow an engineering mindset: 
   - Use systematic, iterative refinement instead of single, analytical fixes. 
   - Focus on empirical evidence, not theoretical assumptions.
2. Focus only on key metrics (e.g., policy performance) to simplify decision-making.
3. Do in-depth, fact-based analysis: 
   - Isolate variables through controlled experiments.
   - Focus only on facts. Seek the simplest explanation through first principles.
4. Seek the most straightforward solutions:
   - Implement direct fixes, not indirect, complex changes.
   (e.g., use shape constraints to fix negative bid prices instead of a new network architecture) 

We have made a note on how to tune the hyperparameters. Please refer to [`hyperparameter-tuning.pdf`](hyperparameter-tuning-note/hyperparameter-tuning.pdf).

## Installation

You can install the TorchBSDE Solver in three ways:

### 1. Direct Use

For a quick installation, download the repository and create a conda environment using:

```bash
conda env create -f environment.yml
```

### 2. As a Git Submodule

Add TorchBSDE as a git submodule to your project:

```bash
git submodule add https://github.com/steve-shao/TorchBSDE-Package.git
git submodule update --init --recursive
```

### 3. Install as a Python Package from GitHub

Install TorchBSDE directly from GitHub using pip:

```bash
pip install git+https://github.com/steve-shao/TorchBSDE-Package.git
```


## Example Usage

```
python -m tests.run_from_config --config_path=configs/hjb_lq_d100.json
```



<br><br><br>

---
---
*Note: Everything below was copied from Jiequn Han's original GitHub repository's README.*

<br><br><br>



## Problems

`equation.py` and `config.py` now support the following problems:

Three examples in ref [1]:
* `HJBLQ`: Hamilton-Jacobi-Bellman (HJB) equation.
* `AllenCahn`: Allen-Cahn equation with a cubic nonlinearity.
* `PricingDefaultRisk`: Nonlinear Black-Scholes equation with default risk in consideration.


Four examples in ref [2]:
* `PricingDiffRate`: Nonlinear Black-Scholes equation for the pricing of European financial derivatives
with different interest rates for borrowing and lending.
* `BurgersType`: Multidimensional Burgers-type PDEs with explicit solution.
* `QuadraticGradient`: An example PDE with quadratically growing derivatives and an explicit solution.
* `ReactionDiffusion`: Time-dependent reaction-diffusion-type example PDE with oscillating explicit solutions.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. Note that the generator function 
and terminal function should be TensorFlow operations while the sample function
can be python operation. A proper config is needed as well.


## Dependencies

Please be aware that the code may not be compatible with the latest version of TensorFlow.

For those using older versions, a version of the deep BSDE solver that is compatible with TensorFlow 1.12 and Python 2 can be found in commit 9d4e332.



## Reference
[1] Han, J., Jentzen, A., and E, W. Overcoming the curse of dimensionality: Solving high-dimensional partial differential equations using deep learning,
<em>Proceedings of the National Academy of Sciences</em>, 115(34), 8505-8510 (2018). [[journal]](https://doi.org/10.1073/pnas.1718942115) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
[2] E, W., Han, J., and Jentzen, A. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations,
<em>Communications in Mathematics and Statistics</em>, 5, 349–380 (2017). 
[[journal]](https://doi.org/10.1007/s40304-017-0117-6) [[arXiv]](https://arxiv.org/abs/1706.04702)

## Citation
```bibtex
@article{HanArnulfE2018solving,
  title={Solving high-dimensional partial differential equations using deep learning},
  author={Han, Jiequn and Jentzen, Arnulf and E, Weinan},
  journal={Proceedings of the National Academy of Sciences},
  volume={115},
  number={34},
  pages={8505--8510},
  year={2018},
  publisher={National Acad Sciences},
  url={https://doi.org/10.1073/pnas.1718942115}
}

@article{EHanArnulf2017deep,
  author={E, Weinan and Han, Jiequn and Jentzen, Arnulf},
  title={Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations},
  journal={Communications in mathematics and statistics},
  volume={5},
  number={4},
  pages={349--380},
  year={2017},
  publisher={Springer},
  url={https://doi.org/10.1007/s40304-017-0117-6}
}
```

