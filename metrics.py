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
