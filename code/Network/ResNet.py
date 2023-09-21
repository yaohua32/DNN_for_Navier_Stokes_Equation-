# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:36:42 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:36:42 
#  */
import numpy as np
import torch
import torch.nn as nn

class Network_sigmoid(nn.Module):
    '''
    '''
    def __init__(self, d_xin:int=1, d_tin:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=3, **kwargs):
        super(Network_sigmoid, self).__init__()
        #
        self.activation = torch.nn.Sigmoid()
        self.fc_in = nn.Linear(d_xin+d_tin, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_xin)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_xin)
            self.ub = torch.ones(1, d_xin)

    def forward(self, x, t):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        xt = torch.cat([x,t], dim=1)
        xt = self.activation(self.fc_in(xt))
        ############################
        for fc_hidden in self.fc_hidden_list:
            xt = self.activation(fc_hidden(xt)) + xt

        return self.fc_out(xt)

class Network_tanh_sin(nn.Module):
    '''
    '''
    def __init__(self, d_xin:int=1, d_tin:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=3, **kwargs):
        super(Network_tanh_sin, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_xin+d_tin, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_xin)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_xin)
            self.ub = torch.ones(1, d_xin)

    def fun_sin(self, xt):
        '''
        '''
        return torch.sin(np.pi * (xt+1.))

    def forward(self, x, t):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        xt = torch.cat([x,t], dim=1)
        xt = self.fc_in(xt)
        ############################
        for fc_hidden in self.fc_hidden_list:
            xt = self.fun_sin(xt)
            xt = self.activation(fc_hidden(xt)) + xt

        return self.fc_out(xt)
    
class Network_tanh(nn.Module):
    '''
    '''
    def __init__(self, d_xin:int=1, d_tin:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=3, **kwargs):
        super(Network_tanh, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_xin+d_tin, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_xin)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_xin)
            self.ub = torch.ones(1, d_xin)

    def forward(self, x, t):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        xt = torch.cat([x,t], dim=1)
        xt = self.activation(self.fc_in(xt))
        ############################
        for fc_hidden in self.fc_hidden_list:
            xt = self.activation(fc_hidden(xt)) + xt

        return self.fc_out(xt)
    
class Model():
    '''
    '''
    def __init__(self, model_type:str, device=None, dtype:torch.dtype=torch.float64):
        self.model_type = model_type
        self.device = device
        torch.set_default_dtype(dtype)
    
    def get_model(self, d_xin:int=1, d_tin:int=1, d_out:int=1, 
                  h_size:int=200, l_size:int=3, **kwargs):
        if self.model_type=='tanh':
            return Network_tanh(d_xin=d_xin, d_tin= d_tin, 
                                d_out=d_out, hidden_size=h_size, 
                                hidden_layers=l_size,
                                **kwargs).to(self.device)
        elif self.model_type=='tanh_sin':
            return Network_tanh_sin(d_xin=d_xin, d_tin= d_tin, 
                                    d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=l_size,
                                    **kwargs).to(self.device)
        elif self.model_type=='sigmoid':
            return Network_sigmoid(d_xin=d_xin, d_tin= d_tin, 
                                    d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=l_size,
                                    **kwargs).to(self.device)
        else:
            raise NotImplementedError(f'Network model: {self.model_type} was not implemented.')