
# CMOBO
Contains experiment codes for CMOBO over the test functions mentioned in the paper.

## Environment

Experiments were conducted in `Python 3.10.6`. See `hardware.txt` for detialed hardware settings.

The Python packages used are:
- BoTorch
- Torch
- numpy
- gpytorch
- platypus
- scipy
- GAUCHE

The details have been included in the `requirements.txt` file, and can be installed with the following command:

```shell
pip install -r requirements.txt
```

For `platypus`, see  [platypus](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus).

## Structure

The repository is structured as follows. The `CMOBO/experiments_CMOBO` directory contains the implementation for the corresponding benchmarks. The `CMOBO/experiments_non_constrained` directory contains the non-constrained version of two baselines. `CMOBO/datasets` contains the necessary data files.

```shell
CMOBO/
├── datasets/
│   ├── caco2++.ipynb
│   ├── caco_domain.pt
│   ├── caco_target.pt
│   ├── domain_ESOL.pt
│   ├── ESOL.ipynb
│   ├── target_ESOL.pt
│   └── TDC.py
├── experiments_CMOBO/
│   ├── branin-currin_CMOBO_exp.ipynb
│   ├── C2-DTLZ2_CMOBO_exp.ipynb
│   ├── caco-2++_CMOBO_exp.ipynb
│   ├── disc_brake_design_CMOBO_exp.ipynb
│   ├── ESOL+_CMOBO_exp.ipynb
│   ├── penicillin_CMOBO_exp.ipynb
│   └── toy_CMOBO_exp.ipynb
├── experiments_non_constrained/
│   ├── qNEHVI_peni_unconstrained.ipynb
│   └── qNEHVI_toy_unconstrained.ipynb
├── README.md
├── requirements.txt
├── hardware.txt
└── toolkits/
    ├── Customized_Kernels.py
    ├── metrics.py
    └── peni.py
```
## toolkits

the toolkits file contains  `peni.py`, `design.py` and `metrics.py`

- `peni.py` and `design.py` are numpy implementations of the Penicillin Function and the Disc Brake Design Problem.
- `metrics.py` contains the implementations of the performance metrics mentioned in the paper.
