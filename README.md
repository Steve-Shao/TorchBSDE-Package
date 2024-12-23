# [Deep BSDE Solver](https://doi.org/10.1073/pnas.1718942115) in PyTorch

## Quick Installation

For a quick installation, you can create a conda environment for Python using the following command:

```bash
conda env create -f environment.yml
conda activate torchbsde
```

## Training

```bash
python main.py --config_path=configs/hjb_lq_d100.json
python -m torchbsde.subnet
```


please add comprehensive docstrings and comments to the code to enhance the readability and maintainability. reorganize the code to make it more readable and modular if necessary, but please make sure to keep the change to the code logic very minimal.

i will modify this script to use pytorch instead of tensorflow. before doing that, to do this careful enough and keep the code logic exactly identifcal, i want to first update the "if main" testing code, so that it comprehensively shows all potential changing details. could you please help me with the testing code? 

use pytorch (instead of tensorflow) throughout this script, while make sure to keep the change in code and logic minimal. 

use pytorch (instead of tensorflow) throughout this script, while make sure to keep the change in code and logic minimal. note that the functions _tf need to be renamed as _torch. 

get rid of `munch` package systematically and throughoutly. instead, just use config as a plain dict. be very careful not to change any code logic. 

get rid of `munch` package systematically and throughoutly (previously, the config of equation is a "munchified" dict.). instead, just use config as a plain dict. be very careful not to change any code logic. 