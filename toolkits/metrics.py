import torch

# Import necessary functionality for hypervolume computation
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

###----------Simple Hypervolume Computation----------###
def HV(Y, ref):
    """
    Computes the hypervolume (HV) of a set of points `Y` with respect to a reference point `ref`.

    Args:
        Y (torch.Tensor): A tensor of objective values, where each row corresponds to a solution.
        ref (torch.Tensor): A tensor representing the reference point in objective space.

    Returns:
        float: The computed hypervolume.
    """
    # Partition the objective space into dominated regions using the reference point
    partition = DominatedPartitioning(ref_point=ref, Y=Y)
    # Compute and return the hypervolume
    hv = partition.compute_hypervolume().item()
    return hv

###----------Simple Violation Computation----------###
def violation(Y, ref):
    """
    Computes the violation values for a set of points `Y` based on a threshold `ref`.

    Args:
        Y (torch.Tensor): A tensor of objective values.
        ref (torch.Tensor): A tensor of threshold values for each objective.

    Returns:
        torch.Tensor: A tensor of violation values, where higher values indicate greater constraint violation.
    """
    # Compute raw violations: values below the threshold
    vio_raw = -torch.min(Y - ref, torch.zeros_like(Y))
    # Sum violations across all objectives
    return vio_raw.sum(-1)

###----------Cumulative Hypervolume Regret----------###
def cum_regret(hv, ub=10000):
    """
    Computes the cumulative hypervolume regret.

    Args:
        hv (torch.Tensor): A tensor of hypervolume values over steps.
        ub (float): The upper bound on the hypervolume value (maximum achievable hypervolume).

    Returns:
        torch.Tensor: A tensor of cumulative hypervolume regret values over steps.
    """
    # Compute regret as the difference from the upper bound
    result = ub - hv.clone()
    # Perform cumulative summation along each dimension
    for dim in range(hv.dim()):
        result = result.cumsum(dim)
    return result

###----------Cumulative Violation Computation----------###
def cum_violation(vio):
    """
    Computes the cumulative violation values.

    Args:
        vio (torch.Tensor): A tensor of violation values over steps.

    Returns:
        torch.Tensor: A tensor of cumulative violation values over steps.
    """
    # Clone the input tensor to preserve original data
    result = vio.clone()
    # Perform cumulative summation along each dimension
    for dim in range(vio.dim()):
        result = result.cumsum(dim)
    return result

###----------Constraint Regret Computation----------###
def constraint_regret(vio, hv, ub):
    """
    Computes the constraint regret, combining hypervolume regret and violation terms.

    Args:
        vio (torch.Tensor): A tensor of violation values.
        hv (torch.Tensor): A tensor of hypervolume values.
        ub (float): The upper bound on the hypervolume value.

    Returns:
        torch.Tensor: A tensor of constraint regret values.
    """
    # Compute hypervolume regret by subtracting from the upper bound
    hv = ub - hv
    # Normalize hypervolume regret to [0, 1]
    hv = hv / hv.max()
    # Normalize violations to [0, 1]
    vio = vio / vio.max()
    # Sum the normalized hypervolume regret and violation values
    Sum = hv + vio
    # Compute cumulative minimum along each dimension
    for dim in range(Sum.dim()):
        Sum = torch.minimum(Sum, Sum.cummin(dim=dim).values)
    return Sum
