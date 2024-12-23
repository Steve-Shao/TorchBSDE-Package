# [DeepBSDE Solver](https://doi.org/10.1073/pnas.1718942115) Package in PyTorch

This repository implements Jiequn Han's [DeepBSDE solver](https://github.com/frankhan91/DeepBSDE), which solves high-dimensional PDEs using deep learning, in PyTorch. 

The code has been restructured to work either as an installable Python package or as a git submodule in other projects. Extensive comments and docstrings have been added to enhance readability and understanding of the implementation. 


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


## Training

```
python -m tests.solve_equation --config_path=configs/hjb_lq_d100.json
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve. 
There are seven PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.


<br><br><br>

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

