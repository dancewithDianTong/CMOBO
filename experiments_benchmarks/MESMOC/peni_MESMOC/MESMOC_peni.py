# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import numpy as np
from model import GaussianProcess
from singlemes import MaxvalueEntropySearch
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real
import sobol_seq
import warnings
import torch
warnings.filterwarnings('ignore')


######################Algorithm input##############################
paths=''
from benchmark_functions import peni

functions_and_contraints= peni
###sample filter###
def voxel_grid_sampling_with_indices(points, voxel_size=5.0):
    # Calculate the minimum and maximum coordinates
    min_coords = np.min(points, axis=0)
    
    # Shift points so that the minimum coordinates are at the origin
    shifted_points = points - min_coords

    # Quantize the points to voxel grid coordinates
    voxel_indices = np.floor(shifted_points / voxel_size).astype(int)

    # Use a dictionary to store unique voxel indices and the corresponding row index
    voxel_dict = {}
    for idx, voxel_idx in enumerate(voxel_indices):
        voxel_idx_tuple = tuple(voxel_idx)
        if voxel_idx_tuple not in voxel_dict:
            voxel_dict[voxel_idx_tuple] = idx

    # Extract the row indices of the sampled points
    sampled_indices = np.array(list(voxel_dict.values()))

    return sampled_indices


M= 3
C= 3
d= 7 
random_seeds = [83810, 14592, 3278, 97196, 36048, 32098, 29256, 18289, 96530, 13434, 88696, 97080, 71482, 11395, 77397, 55302, 4165, 3905, 12280, 28657, 30495, 66237, 78907, 3478, 73563,
26062, 93850, 85181, 91924, 71426, 54987, 28893, 58878, 77236, 36463, 851, 99458, 20926, 91506, 55392, 44597, 36421, 20379, 28221, 44118, 13396, 12156, 49797, 12676, 47052]
for iter in range(10):
    print(f'round {iter+1}')
    np.random.seed(random_seeds[iter])
    total_iterations= 70
    intial_number=10
    sample_number=1
    bound = [0,1]
    Fun_bounds = [bound] * d
    # lower_bounds = np.array([60, 10, 293, 10, 0.01, 600, 5])
    # upper_bounds = np.array([120, 18, 303, 18, 0.1, 700, 6.5])
    # grid = sobol_seq.i4_sobol_generate(d,60,np.random.randint(0,1000))
    initializer = torch.load(f'train_domain_{iter+1}.pt').cpu().numpy()
    # train_y = np.zeros((64,3))
    # for i in range(64):
    #     train_y[i,:] = peni(grid[i,:],d)[:3]
    # ind = voxel_grid_sampling_with_indices(train_y)
    # initializer = grid[ind,:]
    ###################GP Initialisation##########################
    GPs=[]
    Multiplemes=[]
    GPs_C=[]
    Multiplemes_C=[]
    for i in range(M):
        GPs.append(GaussianProcess(d))
    for i in range(C):
        GPs_C.append(GaussianProcess(d))
    #
    for k in range(initializer.shape[0]):
        # exist=True
        # while exist:
        #     design_index = np.random.randint(0, grid.shape[0])
        #     x_rand=list(grid[design_index : (design_index + 1), :][0])
        #     if (any((x_rand == x).all() for x in GPs[0].xValues))==False:
        #         exist=False
        x = initializer[k,:]
        functions_contraints_values=functions_and_contraints(x,d)
        for i in range(M):
            GPs[i].addSample(np.asarray(x),functions_contraints_values[i])
        for i in range(C):
            GPs_C[i].addSample(np.asarray(x),functions_contraints_values[i+M])
        with  open(os.path.join(paths,f'Inputs_{iter+1}.txt'), "a") as filehandle:  
            for item in x:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with  open(os.path.join(paths,f'Outputs_{iter+1}.txt'), "a") as filehandle:  
            for listitem in functions_contraints_values:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()
            
        
    for i in range(M):   
        GPs[i].fitModel()
        Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
    for i in range(C):   
        GPs_C[i].fitModel()
        Multiplemes_C.append(MaxvalueEntropySearch(GPs_C[i]))
        

    for l in range(total_iterations):
        print('*-')
        for i in range(M):
            Multiplemes[i]=MaxvalueEntropySearch(GPs[i])
            Multiplemes[i].Sampling_RFM()
        for i in range(C):
            Multiplemes_C[i]=MaxvalueEntropySearch(GPs_C[i])
            Multiplemes_C[i].Sampling_RFM()
        max_samples=[]
        max_samples_constraints=[]
        for j in range(sample_number):
            for i in range(M):
                Multiplemes[i].weigh_sampling()
            for i in range(C):
                Multiplemes_C[i].weigh_sampling()
            cheap_pareto_front=[]
            def CMO(xi):
                xi=np.asarray(xi)
                y=[Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
                y_c=[Multiplemes_C[i].f_regression(xi)[0][0] for i in range(len(GPs_C))]
                return y,y_c
            
            problem = Problem(d, M,C)
            problem.types[:] = Real(bound[0], bound[1])

            problem.constraints[:]=[">=0" for i in range(C)]
            problem.function = CMO
            algorithm = NSGAII(problem)
            algorithm.run(1500)

            cheap_pareto_front=[list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values=[list(solution.constraints) for solution in algorithm.result]
            
            maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]#this is picking the max over the pareto: best case
            maxofconstraints=[-1*min(f) for f in list(zip(*cheap_constraints_values))]
            max_samples.append(maxoffunctions)
            max_samples_constraints.append(maxofconstraints)
        def mesmo_acq(x):
            
            if np.prod([GPs_C[i].getmeanPrediction(x)>=0 for i in range(len(GPs_C))]):
                multi_obj_acq_total=0
                for j in range(sample_number):
                    multi_obj_acq_sample=0
                    for i in range(M):
                        multi_obj_acq_sample=multi_obj_acq_sample+Multiplemes[i].single_acq(np.asarray(x),max_samples[j][i])
                    for i in range(C):
                        multi_obj_acq_sample=multi_obj_acq_sample+Multiplemes_C[i].single_acq(np.asarray(x),max_samples_constraints[j][i])
                    multi_obj_acq_total=multi_obj_acq_total+multi_obj_acq_sample
                return (multi_obj_acq_total/sample_number)
            else:
                return 10e10


            
        # l-bfgs-b acquisation optimization
        x_tries = np.random.uniform(bound[0], bound[1],size=(1000, d))
        y_tries=[mesmo_acq(x) for x in x_tries]
        sorted_indecies=np.argsort(y_tries)
        i=0
        x_best=x_tries[sorted_indecies[i]]
        while (any((x_best == x).all() for x in GPs[0].xValues)):
            print(x_best)
            print(GPs[0].xValues)
            i=i+1
            x_best=x_tries[sorted_indecies[i]]
        y_best=y_tries[sorted_indecies[i]]
        x_seed=list(np.random.uniform(low=bound[0], high=bound[1], size=(100,d)))    
        for x_try in x_seed:
            result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=Fun_bounds)
            if not result.success:
                continue
            if ((result.fun<=y_best) and (not (result.x in np.asarray(GPs[0].xValues)))):
                x_best=result.x
                y_best=result.fun
                
            

    #---------------Updating and fitting the GPs-----------------  
        functions_contraints_values=functions_and_contraints(x_best,d)
        for i in range(M): 
            GPs[i].addSample(np.asarray(x_best),functions_contraints_values[i])
            GPs[i].fitModel()
        for i in range(C): 

            GPs_C[i].addSample(x_best,functions_contraints_values[M+i])
            GPs_C[i].fitModel()
        with  open(os.path.join(paths,f'Inputs_{iter+1}.txt'), "a") as filehandle:  
            for item in x_best:
                filehandle.write('%f ' % item)
            filehandle.write('\n')
        filehandle.close()
        with  open(os.path.join(paths,f'Outputs_{iter+1}.txt'), "a") as filehandle:  
            for listitem in functions_contraints_values:
                filehandle.write('%f ' % listitem)
            filehandle.write('\n')
        filehandle.close()
    print('\n')

