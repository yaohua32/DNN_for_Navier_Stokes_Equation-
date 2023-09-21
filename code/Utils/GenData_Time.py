# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:42:27 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:42:27 
#  */
import numpy as np 
import torch 
import os 
import sys 
from scipy.stats import qmc
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from Problems.Module_Time import Problem

class GenData():
    '''
    '''
    def __init__(self, problem:Problem, dtype:np.dtype=np.float32):
        self.problem = problem
        self.dtype = dtype
        self.lhs_t = qmc.LatinHypercube(1)
        self.lhs_x = qmc.LatinHypercube(self.problem.dim)

    def get_in(self, Nx_size:int, Nt_size:int, method:str='hypercube')->torch.tensor:
        '''
        '''
        if method=='mesh':
            t = np.linspace(self.problem._t0, self.problem._tT, Nt_size).reshape(-1,1)  
        elif method=='random':
            t = np.random.uniform(self.problem._t0, self.problem._tT, [Nt_size,1])  
        elif method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), self.problem._t0, self.problem._tT)
        else:
            raise NotImplementedError
        x = qmc.scale(self.lhs_x.random(Nx_size*Nt_size), self.problem._lb, self.problem._ub)
        t = t.repeat(Nx_size, axis=0)
        
        return torch.tensor(x.astype(self.dtype)), torch.tensor(t.astype(self.dtype))
    
    def get_bd(self, N_bd_each_face:int, Nt_size:int, method='hypercube')->torch.tensor:
        '''
        '''
        x_list = []
        if method=='mesh':
            t = np.linspace(self.problem._t0, self.problem._tT, Nt_size).reshape(-1,1)
        elif method=='random':
            t = np.random.uniform(self.problem._t0, self.problem._tT, [Nt_size,1])  
        elif method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), self.problem._t0, self.problem._tT)
        else:
            raise NotImplementedError
        x = qmc.scale(self.lhs_x.random(N_bd_each_face * Nt_size), self.problem._lb, self.problem._ub)
        for d in range(self.problem.dim):
            x_lb, x_ub= np.copy(x), np.copy(x)
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self.problem._lb[d], self.problem._ub[d]
            x_list.extend([torch.from_numpy(x_lb.astype(self.dtype)), 
                            torch.from_numpy(x_ub.astype(self.dtype))])
        t = t.repeat(N_bd_each_face, axis=0)

        return x_list, torch.from_numpy(t.astype(self.dtype))

    def get_init(self, Nx_init:int, given_t:float=None)->torch.tensor:
        '''
        '''
        x = qmc.scale(self.lhs_x.random(Nx_init), self.problem._lb, self.problem._ub)
        if given_t is not None:
            t = given_t * np.ones([Nx_init, 1])
        else:
            t = self.problem._t0 * np.ones([Nx_init, 1])
        #
        return torch.from_numpy(x.astype(self.dtype)), torch.from_numpy(t.astype(self.dtype))
    
    def get_txc(self, N_xc:int, Nt_size:int, R_max:float=1e-3, R_min:float=1e-8, 
                R_method:str='R_first', t_method:str='hypercube')->torch.tensor:
        '''
        '''
        if R_max<R_min:
            raise ValueError('R_max should be larger than R_min.')
        elif (2.*R_max)>np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        #
        R = np.random.uniform(R_min, R_max, [N_xc*Nt_size, 1])
        lb, ub = self.problem._lb + R, self.problem._ub - R
        #
        if R_method=='R_first':
            if t_method=='hypercube':
                t = qmc.scale(self.lhs_t.random(Nt_size), self.problem._t0, self.problem._tT)
            elif t_method=='mesh':
                t = np.linspace(self.problem._t0, self.problem._tT, Nt_size).reshape(-1,1)
            else:
                raise NotImplementedError
            xc = self.lhs_x.random(N_xc*Nt_size) * (ub - lb) + lb 
            t = t.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError
        
        return torch.tensor(R.astype(self.dtype)).view(-1, 1, 1),\
            torch.tensor(xc.astype(self.dtype)).view(-1,1,self.problem.dim),\
                torch.tensor(t.astype(self.dtype)).view(-1, 1, 1)
    
    def get_x_scaled(self, Nx_scaled, method:str='mesh')->torch.tensor:
        '''
        '''
        if method=='random':
            if self.problem.dim==1:
                x_scaled = np.random.uniform(-1., 1., size=(Nx_scaled, self.problem.dim))
            else:
                X_d = np.random.normal(size=(Nx_scaled, self.problem.dim+2))#生成m个(d+2)维空间中的坐标               
                X_d = X_d / np.sqrt(np.sum(X_d**2, axis=1, keepdims=True))  #对这个坐标进行单位化，得到(d+2)维球面上的点   
                #                        
                x_scaled = X_d[:,0:self.problem.dim].reshape([-1, self.problem.dim])
        elif method=='hypercube':
            if self.problem.dim==1:
                x_scaled = qmc.scale(self.lhs_x.random(Nx_scaled), -1., 1.)
            else:
                X_d = qmc.scale(self.lhs_x.random(Nx_scaled), -1., 1.)
                # 
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) < 1.)[0]
                x_scaled = X_d[index,:]
        elif method=='mesh':
            if self.problem.dim==1:
                x_scaled = np.linspace(-1., 1., Nx_scaled).reshape(-1, self.problem.dim)
            elif self.problem.dim==2:
                x, y = np.meshgrid(np.linspace(-1., 1., Nx_scaled), np.linspace(-1., 1., Nx_scaled))
                X_d = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_scaled = X_d[index,:]
            else:
                raise NotImplementedError('The mesh method is not availabel for d>3.')
        else:
            raise NotImplementedError
            
        return torch.tensor(x_scaled.astype(self.dtype))

    def get_t_scaled(self, Nt_scaled, method:str='mesh')->torch.tensor:
        '''
        '''
        if method=='random':
            t_scaled = np.random.uniform(-1., 1., size=(Nt_scaled, 1))
        elif method=='hypercube':
            t_scaled = qmc.scale(self.lhs_t.random(Nt_scaled), -1., 1.)
        elif method=='mesh':
            t_scaled = np.linspace(-1., 1., Nt_scaled).reshape(-1, 1)
        else:
            raise NotImplementedError
            
        return torch.tensor(t_scaled.astype(self.dtype))