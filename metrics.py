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

def cum_regret(tensor, ub = 10000):
    # Initialize the result tensor with the original tensor
    result = tensor.clone()
    result = ub - result
    
    # Perform cumulative sum along each dimension
    for dim in range(tensor.dim()):
        result = result.cumsum(dim)
    return result
###Cumulative violation###
def cum_violation(tensor, ref):
    result = tensor.clone()
    vio = violation(result, ref)
    for dim in range(tensor.dim()):
        vio = vio.cumsum(dim)
    return vio
###constraint regret###
def constraint_regret(Y, ref, hv):
    vio = violation(Y, ref)
    hv = 10000 - hv
    hv = (hv - hv.min()) / (hv.max() - hv.min())
    vio = (vio - vio.min()) / (vio.max() - vio.min())
    Sum = hv + vio
    for dim in range(Sum.dim()):
        Sum = torch.minimum(Sum, Sum.cummin(dim=dim).values)
    return Sum

