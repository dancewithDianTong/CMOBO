# CMOBO
CMOBO_code

## peni.py
contains a function that simulates the biological process of penicillin reaction, which has been used as a test function for MOBO in *Scalable Bayesian Optimization Accelerates Process Optimization of Penicillin Production, 2021*(https://openreview.net/forum?id=UVdSYXMNdOe). It was first released in *A modular simulation package for fed-batch fermentation: penicillin production, 2002*(https://www.sciencedirect.com/science/article/abs/pii/S0098135402001278)

it takes a 7-D input and its output is 3-D.

it could also be imported in botorch directly

## CMOBO.ipynb
I've written a simple botorch work flow for CMOBO in discrete case.

## Caco-2 plus
The dataset menstioned in BOtied: Multi-objective Bayesian optimization with tied multivariate ranks,
an application to active drug discovery(https://icml-compbio.github.io/2023/papers/WCBICML2023_paper71.pdf)

It can be found in: https://github.com/jiwoncpark/botied/blob/main/botied/objectives/caco2_plus.py
