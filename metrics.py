import torch

###----------hypervolume----------###
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

def HV(Y, ref):
    """
    input:
    Y: a (t by m) tensor, current observed objective vlaues. 
    ref: a (m) dimensional tensor

    return:
    hv: the hypervolume indicator of current observation
    """
    partition = DominatedPartitioning(ref_point=ref, Y=Y)
    hv = partition.compute_hypervolume().item()
    return hv

###----------violation----------###
def violation(Y, ref):
    vio_raw = - torch.min(Y -ref, torch.zeros_like(Y))
    return (vio_raw).sum(-1)

###Cumulative regret##

def cum_regret(hv, ub = 10000):
    # Initialize the result tensor with the original tensor
    result = hv.clone()
    result = ub - result
    
    # Perform cumulative sum along each dimension
    for dim in range(hv.dim()):
        result = result.cumsum(dim)
    return result
###Cumulative violation###
def cum_violation(vio):
    result = vio.clone()
    for dim in range(vio.dim()):
        result = result.cumsum(dim)
    return result
###constraint regret###
def constraint_regret(vio,  hv, ub):
    hv = ub - hv
    hv = (hv) / (hv.max())
    vio = (vio) / (vio.max())
    Sum = hv + vio
    for dim in range(Sum.dim()):
        Sum = torch.minimum(Sum, Sum.cummin(dim=dim).values)
    return Sum

