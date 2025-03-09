# CMOBOO

**Constrained Multi-objective Bayesian Optimization through Optimistic Constraints Estimation**  
This repository contains the experimental code for CMOBOO, as described in the paper:

> Li, Diantong, Fengxue Zhang, Chong Liu, and Yuxin Chen. “Constrained Multi-objective Bayesian Optimization through Optimistic Constraints Estimation.” *arXiv preprint arXiv:2411.03641*, 2024.

If you use this code in your research, please consider citing the paper (see the Citation section below).

---

## Overview

CMOBO is designed to perform constrained multi-objective Bayesian optimization on several benchmark functions. The repository includes implementations for both constrained experiments and non-constrained baseline comparisons. Detailed instructions, data files, and experiment notebooks are provided.

---

## Citation

If you use this repository, please cite the paper using the following BibTeX entry:

```bibtex
@article{li2024constrained,
  title={Constrained Multi-objective Bayesian Optimization through Optimistic Constraints Estimation},
  author={Li, Diantong and Zhang, Fengxue and Liu, Chong and Chen, Yuxin},
  journal={arXiv preprint arXiv:2411.03641},
  year={2024}
}
```

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

```bash
pip install -r requirements.txt
```

For `platypus`, see  [platypus](https://platypus.readthedocs.io/en/latest/getting-started.html#installing-platypus).

## Structure

(1) Clone the Repository:

```bash
git clone https://github.com/yourusername/CMOBO.git
cd CMOBO
```

(2) Install Dependencies:

```bash
pip install -r requirements.txt
```

(3) Run an Experiment Notebook:

Open one of the Jupyter notebooks in the experiments_CMOBO directory (e.g., toy_CMOBO_exp.ipynb) using your preferred Jupyter environment.

## Repository Structure

The repository is organized as follows:

``` bash
CMOBO/
├── experiments_CMOBO/
│   ├── branin-currin_CMOBO_exp.ipynb    # Jupyter notebook for the Branin-Currin experiment
│   ├── C2-DTLZ2_CMOBO_exp.ipynb         # Jupyter notebook for the C2-DTLZ2 experiment
│   ├── caco-2++_CMOBO_exp.ipynb         # Jupyter notebook for the Caco-2++ experiment
│   ├── disc_brake_design_CMOBO_exp.ipynb  # Jupyter notebook for the Disc Brake Design experiment
│   ├── ESOL+_CMOBO_exp.ipynb             # Jupyter notebook for the ESOL+ experiment
│   ├── penicillin_CMOBO_exp.ipynb         # Jupyter notebook for the Penicillin experiment
│   └── toy_CMOBO_exp.ipynb                # Jupyter notebook for the Toy function experiment
├── experiments_non_constrained/
│   ├── qNEHVI_peni_unconstrained.ipynb    # Notebook for the unconstrained qNEHVI experiment (Penicillin)
│   └── qNEHVI_toy_unconstrained.ipynb     # Notebook for the unconstrained qNEHVI experiment (Toy function)
├── experiments_benchmarks/
│   ├── MESMOC/                           # Contains Python scripts and text files for MESMOC benchmarks
│   │   └── ...                           # (Python files such as `MESMOC_branin_currin.py`, `model.py`, etc.)
│   ├── qNEHVI/                           # Contains Jupyter notebooks for qNEHVI benchmarks
│   │   └── ...                           # (e.g. `branin_currin_qNEHVI.ipynb`, `caco2++_qNEHVI.ipynb`, etc.)
│   └── qParEGO/                          # Contains Jupyter notebooks for qParEGO benchmarks
│       └── ...                           # (e.g. `qParEGO_branin_currin.ipynb`, etc.)
├── toolkits/
│   ├── Customized_Kernels.py             # Python module for custom GP kernels
│   ├── design.py                         # Python module for design-related functions
│   ├── metrics.py                        # Python module for performance metrics
│   └── peni.py                           # Python module with helper functions for penicillin experiments
├── README.md                             # Project readme
├── requirements.txt                      # Python dependency list
```

## toolkits

the toolkits file contains  `peni.py`, `design.py` and `metrics.py`

- `peni.py` and `design.py` are numpy implementations of the Penicillin Function and the Disc Brake Design Problem.
- `metrics.py` contains the implementations of the performance metrics mentioned in the paper.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors.
