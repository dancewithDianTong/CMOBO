###Define a Customized RBF Kernel###


from gpytorch.kernels import RBFKernel, ScaleKernel

def Customized_RBF(lengthscale, outputscale):
    rbf_base = RBFKernel()

    #----set the customized length scale here QAQ-----#
    rbf_base.lengthscale = lengthscale
    #----end setting length scale-----#

    covar_module = ScaleKernel(
    base_kernel=rbf_base,
    )
    #----set the customized output scale here QAQ-----#
    covar_module.outputscale = outputscale
    #----end setting output scale-----#
    return covar_module
