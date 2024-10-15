# CMOBO
Constains experiment codes for CMOBO over the test functions mentioned in the paper.

## Environment
(TD: add verified hardwares.)
python package used:
- botorch
- torch
- numpy
- gpytorch
- platypus
- scipy

The details has been included in the requirementx.txt, and could be imported with the following command:

```shell

pip install -r requirements.txt

```

## Structure

The respository is structured as the following. The CMOBO\experimeents_CMOBO directory contains the implementation for the corresponding benchmarks. The CMOBO\experiments_non_constrained containts the non-constrained version of two baselines. CMOBO\datasets contains the necessary data files.

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
│   ├── disc brake design_CMOBO_exp.ipynb
│   ├── ESOL+_CMOBO_exp.ipynb
│   ├── penicillin_CMOBO_exp.ipynb
│   └── toy_CMOBO_exp.ipynb
├── experiments_non_constrained/
│   ├── qNEHVI_peni_unconstrained.ipynb
│   └── qNEHVI_toy_unconstrained.ipynb
├── README.md
└── toolkits/
    ├── Customized_Kernels.py
    ├── metrics.py
    └── peni.py
```