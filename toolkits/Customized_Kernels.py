###Define a Customized RBF Kernel###


from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel

##RBF
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

##Matern
def Customized_Matern(lengthscale, smoothness = 2.5, outputscale=1):
    matern_base = MaternKernel(nu=smoothness)

    #----set the customized length scale here QAQ-----#
    matern_base.lengthscale = lengthscale
    #----end setting length scale-----#

    covar_module = ScaleKernel(
    base_kernel=matern_base,
    )
    #----set the customized output scale here QAQ-----#
    covar_module.outputscale = outputscale
    #----end setting output scale-----#
    return covar_module