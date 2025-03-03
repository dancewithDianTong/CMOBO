# CMOBO

**Constrained Multi-objective Bayesian Optimization through Optimistic Constraints Estimation**  
This repository contains the experimental code for CMOBO, as described in the paper:

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

All dependencies are listed in the requirements.txt file. Install them using:

```bash
pip install -r requirements.txt
```

## Quick Start

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
├── datasets/
│   ├── caco2++.ipynb         # Notebook for Caco-2++ dataset exploration
│   ├── caco_domain.pt        # Domain definition for Caco-2++
│   ├── caco_target.pt        # Target values for Caco-2++
│   ├── domain_ESOL.pt        # Domain definition for ESOL dataset
│   ├── ESOL.ipynb            # Notebook for ESOL dataset exploration
│   ├── target_ESOL.pt        # Target values for ESOL dataset
│   └── TDC.py                # Therapeutics Data Commons helper functions
├── experiments_CMOBO/
│   ├── branin-currin_CMOBO_exp.ipynb    # Branin-Currin benchmark experiment
│   ├── C2-DTLZ2_CMOBO_exp.ipynb         # C2-DTLZ2 benchmark experiment
│   ├── caco-2++_CMOBO_exp.ipynb         # Caco-2++ benchmark experiment
│   ├── disc_brake_design_CMOBO_exp.ipynb  # Disc Brake Design benchmark experiment
│   ├── ESOL+_CMOBO_exp.ipynb             # ESOL+ benchmark experiment
│   ├── penicillin_CMOBO_exp.ipynb         # Penicillin production benchmark experiment
│   └── toy_CMOBO_exp.ipynb                # Toy function benchmark experiment
├── experiments_non_constrained/
│   ├── qNEHVI_peni_unconstrained.ipynb    # Unconstrained qNEHVI experiment (Penicillin)
│   └── qNEHVI_toy_unconstrained.ipynb     # Unconstrained qNEHVI experiment (Toy function)
├── toolkits/
│   ├── Customized_Kernels.py  # Customized kernel implementations for GP models
│   ├── metrics.py             # Performance metrics and evaluation functions
│   └── peni.py                # Helper functions for Penicillin experiments
└── README.md
```

## Usage

- Experiment Notebooks:
Each Jupyter notebook in the experiments_CMOBO and experiments_non_constrained directories is self-contained. Open them in Jupyter Notebook or JupyterLab to run the experiments interactively.

- Data Files:
The datasets directory contains the required data files for running the experiments. Ensure these files are in place when executing the notebooks.

- Toolkits:
The code in the toolkits directory provides auxiliary functions (e.g., custom kernels, evaluation metrics) that support the main experimental code.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors.

