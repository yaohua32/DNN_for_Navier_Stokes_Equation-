# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:39:21 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:39:21 
#  */
import numpy as np 
import torch 
import sys
import os
import math
import h5py
from torch.autograd import Variable
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#
import Problems.Module_Time as Module
from Utils.GenData_Time import GenData

class Problem(Module.Problem):
    '''
    The NS problem
    '''
    def __init__(self,dtype:np.dtype=np.float64, 
                 t0_loc:int=0, tT_loc:int=101):
        self._dim = 2
        self._name = 'NS_unsteady'
        self._t_mesh = np.linspace(0., 30., 101)
        self._t0_loc = t0_loc
        self._tT_loc = tT_loc
        #
        self._t0 = self._t_mesh[t0_loc]
        self._tT = self._t_mesh[tT_loc-1]
        self._lb = np.array([-math.pi, -math.pi])
        self._ub = np.array([math.pi, math.pi])
        self._Re = 1./4.66e-4
        #
        self._dtype = dtype
        self._eps = sys.float_info.epsilon
        self._get_true()
    
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    @property
    def lb(self):
        return np.array([self._lb[0], 
                         self._lb[1]]).reshape(-1, self.dim).astype(self._dtype)

    @property
    def ub(self):
        return np.array([self._ub[0], 
                         self._ub[1]]).reshape(-1, self.dim).astype(self._dtype)
    
    def _get_pred(self):
        '''
        '''
        t_mesh = self._t_mesh[self._t0_loc:self._tT_loc].reshape(-1,1)
        #
        d = 2*np.pi/512
        x = np.linspace(-np.pi, np.pi-d, 512)
        y = np.linspace(-np.pi, np.pi-d, 512)
        x_mesh = np.array([[xi, yi] for xi in x for yi in y])

        return torch.from_numpy(x_mesh.astype(self._dtype)), \
            torch.from_numpy(t_mesh.astype(self._dtype))
    
    def _get_true(self):
        '''
        Obtain the true data from .h5 file:
        Input: None
        Output: t, x, u, v, p
        '''
        f = h5py.File('./data/train/data_16.h5', 'r')
        #
        self.tx = np.zeros((0,3))
        self.u = np.zeros((0,1))
        self.v = np.zeros((0,1))
        self.p = np.zeros((0,1))
        #
        for loc in range(self._t0_loc, self._tT_loc):
            key = list(f.keys())[loc]
            t = float(key)
            x = f[key][:,0:2]
            u = f[key][:,2][:,None]
            v = f[key][:,3][:,None]
            p = f[key][:,4][:,None]
            self.tx = np.concatenate((self.tx, np.concatenate((np.full((x.shape[0],1), t), x), axis=1)), axis=0)
            self.u  = np.concatenate((self.u, u), axis=0)
            self.v  = np.concatenate((self.v, v), axis=0)
            self.p  = np.concatenate((self.p, p), axis=0)
        f.close()

    def _fun_true(self, xs:torch.tensor=None,
               t:torch.tensor=None)->torch.tensor:
        '''
        '''
        if xs is not None:
            raise NotImplementedError('Explicit solution is not implemented.')
        else:
            t = torch.from_numpy(self.tx[:,0:1].astype(self._dtype))
            x = torch.from_numpy(self.tx[:,1:3].astype(self._dtype))
            #
            u = torch.from_numpy(self.u.astype(self._dtype))
            v = torch.from_numpy(self.v.astype(self._dtype))
            p = torch.from_numpy(self.p.astype(self._dtype))

            return u, v, p, x, t

    def fun_bd(self, x_list:list[torch.tensor], t:torch.tensor, 
               model=None)->torch.tensor:
        '''
        '''
        t_list, x_lb_list, x_ub_list = [], [], []
        for d in range(self.dim):
            t_list.append(t)
            x_lb_list.append(x_list[2*d])
            x_ub_list.append(x_list[2*d+1])
        t = torch.cat(t_list, dim=0)
        x_lb = torch.cat(x_lb_list, dim=0)
        x_ub = torch.cat(x_ub_list, dim=0)
        #
        cond_list = []
        #
        x_lb = Variable(x_lb, requires_grad=True)
        x_ub = Variable(x_ub, requires_grad=True)
        #
        f_lb_nn = model(x_lb, t)
        u_lb, v_lb, p_lb = f_lb_nn[:,0:1], f_lb_nn[:,1:2], f_lb_nn[:,2:]
        du_lb = self._grad_u(x_lb, u_lb)
        dv_lb = self._grad_u(x_lb, v_lb)
        #
        f_ub_nn = model(x_ub, t)
        u_ub, v_ub, p_ub = f_ub_nn[:,0:1], f_ub_nn[:,1:2], f_ub_nn[:,2:]
        du_ub = self._grad_u(x_ub, u_ub)
        dv_ub = self._grad_u(x_ub, v_ub)
        #
        cond_list.append(u_lb - u_ub)
        cond_list.append(v_lb - v_ub)
        cond_list.append(du_lb[:,0:1] - du_ub[:,0:1])
        cond_list.append(du_lb[:,1:] - du_ub[:,1:])
        cond_list.append(dv_lb[:,0:1] - dv_ub[:,0:1])
        cond_list.append(dv_lb[:,1:] - dv_ub[:,1:])

        return torch.cat(cond_list, dim=0)

    def fun_f(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        '''
        fx = 0.025 * torch.sin(2*x[:,0:1]) * torch.cos(2*x[:,1:2])
        fy = -0.025 * torch.cos(2*x[:,0:1]) * torch.sin(2*x[:,1:2])

        return fx, fy
    
    def weak(self, model, 
             x_scaled:torch.tensor, xc:torch.tensor,
             t:torch.tensor, R:torch.tensor, 
             phi:torch.tensor, dphi_scaled:torch.tensor
             )->torch.tensor:
        '''
        '''
        ###############
        m = x_scaled.shape[0] 
        xs = x_scaled * R + xc 
        t = t.repeat(1, m, 1) 
        #
        xs = Variable(xs.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        # 
        uvp = model(xs, t)
        u, v, p = uvp[:,0:1], uvp[:,1:2], uvp[:,2:]
        #
        dux, dut= self._grad_u(xs, u), self._grad_u(t, u)
        dvx, dvt = self._grad_u(xs, v), self._grad_u(t, v)
        dpx = self._grad_u(xs, p)
        # 
        u, v, p = u.view(-1,m,1), v.view(-1,m,1), p.view(-1,m,1)
        dux, dut = dux.view(-1,m,self.dim), dut.view(-1,m,1)
        dvx, dvt = dvx.view(-1,m,self.dim), dvt.view(-1,m,1)
        dpx = dpx.view(-1,m,self.dim)
        #
        gx, gy = self.fun_f(xs, t)
        gx, gy = gx.view(-1,m,1), gy.view(-1,m,1)
        # 
        dphi = dphi_scaled/R 
        ######
        eq_u = (torch.mean(torch.sum(dut * phi, dim=2, keepdim=True), dim=1) 
                + torch.mean(torch.sum((u*dux[:,:,0:1] + v*dux[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)
                + (1./self._Re) * torch.mean(torch.sum(dux * dphi, dim=2, keepdim=True),dim=1)
                # + torch.mean(torch.sum( dpx[:,:,0:1] * phi, dim=2, keepdim=True), dim=1)
                - torch.mean(torch.sum( p * dphi[:,:,0:1], dim=2, keepdim=True), dim=1)
                - torch.mean(torch.sum( gx * phi, dim=2, keepdim=True), dim=1))
        eq_v = (torch.mean(torch.sum(dvt * phi, dim=2, keepdim=True), dim=1) 
                + torch.mean(torch.sum((u*dvx[:,:,0:1] + v*dvx[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)
                + (1./self._Re) * torch.mean(torch.sum(dvx * dphi, dim=2, keepdim=True),dim=1)
                # + torch.mean(torch.sum( dpx[:,:,1:2] * phi, dim=2, keepdim=True), dim=1)
                - torch.mean(torch.sum( p * dphi[:,:,1:2], dim=2, keepdim=True), dim=1)
                - torch.mean(torch.sum( gy * phi, dim=2, keepdim=True), dim=1))
        #
        eq_div = torch.mean(torch.sum((dux[:,:,0:1] + dvx[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)

        return eq_u, eq_v, eq_div

    def strong(self, model, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        '''
        ############ 
        xs = Variable(x, requires_grad=True)
        t = Variable(t, requires_grad=True)
        xs_list = torch.split(xs, split_size_or_sections=1, dim=1)
        ############# 
        uvp = model(torch.cat(xs_list, dim=1), t)
        u, v, p = uvp[:,0:1], uvp[:,1:2], uvp[:,2:]
        #
        dux, dut = self._grad_u(xs, u), self._grad_u(t, u)
        dvx, dvt = self._grad_u(xs, v), self._grad_u(t, v)
        dpx = self._grad_u(xs, p)
        #
        Lu, Lv = self._Laplace_u(xs_list, dux), self._Laplace_u(xs_list, dvx)
        #
        gx, gy = self.fun_f(xs, t)
        ##############
        eq_u = dut + u*dux[:,0:1] + v*dux[:,1:] + dpx[:,0:1] - (1./self._Re) * Lu - gx 
        eq_v = dvt + u*dvx[:,0:1] + v*dvx[:,1:] + dpx[:,1:2] - (1./self._Re) * Lv - gy
        eq_div = dux[:,0:1] + dvx[:,1:2]
        
        return eq_u, eq_v, eq_div