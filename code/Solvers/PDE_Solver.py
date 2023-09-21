# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:29:31 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:29:31 
#  */
import numpy as np 
import scipy.io
import time
import os
import h5py
import torch
#
from Network.ResNet import Model
from Utils.Error import Error
from Utils.TestFun import TestFun
from Utils.GenData_Time import GenData
from Problems.Module_Time import Problem
import Solvers.Module as Module
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
try:
    print(f'{torch.cuda.get_device_name(0)}')
except:
    pass

class PDE_Solver(Module.Solver):
    '''
    '''
    def __init__(self, problem:Problem(), loss_type:str,
                 Num_particles:int, Num_tin_size:int, Nx_integral:int, 
                 train_xbd_size_each_face:int, train_tbd_size:int,
                 R_max:float, maxIter:int, 
                 lr:float, net_type:str, **kwargs):
        #
        self.loss_type = loss_type
        self.Num_particles = Num_particles
        self.Num_tin_size = Num_tin_size
        self.Nx_integral = Nx_integral
        self.train_xbd_size_each_face = train_xbd_size_each_face
        self.train_tbd_size = train_tbd_size
        self.Rmax = R_max
        self.iters = maxIter
        self.lr = lr
        self.net_type = net_type
        # Other settings
        self.lrDecay = kwargs['lrDecay']
        self.Rmin = kwargs['R_min']
        self.weight = kwargs['weight']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.dtype = kwargs['dtype']
        #
        self.problem = problem
        self._test_fun = TestFun(kwargs['test_fun'], problem.dim)
        self.data = GenData(self.problem, dtype=self.dtype['numpy'])

    def _save(self, save_path:str, model_type:str)->None:
        '''
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 
        if model_type=='model_final':
            dict_loss = {}
            dict_loss['loss_all'] = self.loss_list
            dict_loss['error_u'] = self.error_u 
            dict_loss['error_v'] = self.error_v
            dict_loss['error_p'] = self.error_p
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+f'loss_error_saved.mat', dict_loss)
        #
        model_dict = {'model':self.model.state_dict()}
        torch.save(model_dict, save_path+f'trained_{model_type}.pth')

    def _load(self, load_path:str, model_type:str)->None:
        '''
        '''
        model_dict = torch.load(load_path+f'trained_{model_type}.pth', 
                                map_location=torch.device(device))
        try:
            self.model.load_state_dict(model_dict['model'])
        except:
            self.get_net() 
            self.model.load_state_dict(model_dict['model'])

    def pred(self, load_path:str, model_type:str)->None:
        '''
        '''
        # load the trained model
        self._load(load_path, model_type)
        ####################################################
        x_mesh, t_mesh = self.problem._get_pred()
        #
        writeH5File = h5py.File(load_path+'Prediction.h5', 'w')
        with torch.no_grad():
            for t_loc in range(len(t_mesh)):
                t = t_mesh[t_loc,:].repeat(512*512,1)
                uvp_pred = self.model(x_mesh.to(device), t.to(device))
                u_pred, v_pred, p_pred = uvp_pred[:,0:1], uvp_pred[:,1:2], uvp_pred[:,2:]
                #
                frame_data = torch.cat([x_mesh.to(device), u_pred, v_pred, 
                                        p_pred], dim=1).view(512,512,5)
                writeH5File.create_dataset('{:0>5.1f}'.format(float(t_mesh[t_loc])), 
                                           data=frame_data.detach().cpu().numpy())
        writeH5File.close()

    def get_net(self)->None:
        '''
        '''
        kwargs = {'d_xin':self.problem.dim,
                  'd_tin': 1,
                  'd_out': 3,
                  'h_size': self.hidden_n,
                  'l_size': self.hidden_l,
                  'lb':torch.from_numpy(self.problem.lb).to(device), 
                  'ub':torch.from_numpy(self.problem.ub).to(device)}
        self.model = Model(self.net_type, device, dtype=self.dtype['torch']).get_model(**kwargs)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr}
            ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=(1. - self.lrDecay/self.iters), last_epoch=-1)

    def get_testFun(self):
        '''
        '''
        x_scaled = self.data.get_x_scaled(Nx_scaled=self.Nx_integral, 
                                          method=self.int_method)
        phi, dphi_scaled, _ = self._test_fun.get_value(x_scaled)

        return x_scaled, phi, dphi_scaled
    
    def get_in(self):
        '''
        '''
        x_in, t_in = self.data.get_in(Nx_size=self.Num_particles, 
                                    Nt_size=self.Num_tin_size)
        
        return x_in, t_in

    def get_loss(self, **args):
        '''
        '''
        loss_total, loss_all = 0., []
        ############ Mis-match of the data
        uvp_pred = self.model(args['x_data'].to(device), args['t_data'].to(device))
        u_pred, v_pred, p_pred = uvp_pred[:,0:1], uvp_pred[:,1:2], uvp_pred[:,2:]
        #
        u_scale = (torch.mean(args['u_data'].to(device)**2) + 1e-8).detach()
        v_scale = (torch.mean(args['v_data'].to(device)**2) + 1e-8).detach()
        p_scale = (torch.mean(args['p_data'].to(device)**2) + 1e-8).detach()
        #
        loss_u = torch.mean( (u_pred - args['u_data'].to(device))**2 ) / u_scale
        loss_v = torch.mean( (v_pred - args['v_data'].to(device))**2 ) / v_scale
        loss_p = torch.mean( (p_pred - args['p_data'].to(device))**2 ) / p_scale
        ########### Residual inside the domain
        if self.loss_type=='weak':
            R, xc, tc= self.data.get_txc(N_xc=self.Num_particles, Nt_size=self.Num_tin_size,
                                         R_max=args['Rmax'], R_min=args['Rmin'])
            #
            eq_u, eq_v, eq_div = self.problem.weak(
                self.model, args['x_scaled'].to(device), xc.to(device), tc.to(device), 
                R.to(device), args['phi'].to(device), args['dphi_scaled'].to(device))
        elif self.loss_type=='strong':
            eq_u, eq_v, eq_div = self.problem.strong(
                self.model, args['x_in'].to(device), args['t_in'].to(device))
        else:
            raise NotImplementedError
        #
        loss_in = (torch.mean( eq_u**2 ) / u_scale 
                   + torch.mean( eq_v**2 ) / v_scale 
                   + torch.mean( eq_div**2  ) / (u_scale+v_scale))
        ############ Residual on the boundary
        x_bd_list, t_bd = self.data.get_bd(N_bd_each_face=self.train_xbd_size_each_face,
                                           Nt_size=self.train_tbd_size) 
        x_bd_list, t_bd = [item.to(device) for item in x_bd_list], t_bd.to(device)
        cond = self.problem.fun_bd(x_bd_list, t_bd, self.model)
        #
        loss_bd = torch.mean( cond**2 ) / (u_scale+v_scale)
        #############
        loss_total = (self.weight['eq'] * loss_in + self.weight['bd'] * loss_bd + 
                      self.weight['u'] * loss_u + self.weight['v'] * loss_v 
                      + self.weight['p'] * loss_p) / 5.
        loss_all = [loss_in.detach(), loss_bd.detach(), loss_u.detach(),
                    loss_v.detach(), loss_p.detach()]

        return loss_total, loss_all

    def train(self, save_path:str, load_path:str=None, model_type:str=None)->None:
        '''
        '''
        t_start = time.time()
        try:
            self._load(load_path=load_path, model_type=model_type)
            print('*********** Started with a trained model ...... ***************')
        except:
            self.get_net()
            print('*********** Started with a new model ...... ***************')
        #
        u_valid, v_valid, p_valid, x_valid, t_valid = self.problem._fun_true()
        # 
        iter = 0
        best_err, best_loss = 1e10, 1e10
        self.time_list = []
        self.loss_list = []
        self.error_u, self.error_v, self.error_p = [], [], []
        #
        if self.loss_type=='weak':
            x_scaled, phi, dphi_scaled = self.get_testFun()
        #
        for iter in range(self.iters):
            if self.loss_type=='weak':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * iter/self.iters
                loss_train, loss_all = self.get_loss(**{'x_scaled':x_scaled, 'phi':phi, 
                                                'dphi_scaled':dphi_scaled, 
                                                'Rmax':R_adaptive, 'Rmin':self.Rmin,
                                                'x_data': x_valid, 't_data':t_valid,
                                                'u_data': u_valid, 'v_data': v_valid,
                                                'p_data': p_valid})
            else:
                x_in, t_in = self.get_in()
                loss_train, loss_all = self.get_loss(**{'x_in':x_in, 't_in':t_in,
                                                'x_data': x_valid, 't_data':t_valid,
                                                'u_data': u_valid, 'v_data': v_valid,
                                                'p_data': p_valid})
            # Save loss and error 
            self.loss_list.append(loss_train.item())
            self.time_list.append(time.time()-t_start)
            with torch.no_grad():
                uvp_pred = self.model(x_valid.to(device), t_valid.to(device))
                u_pred, v_pred, p_pred = uvp_pred[:,0:1], uvp_pred[:,1:2], uvp_pred[:,2:]
                error_u_valid = Error().L2_error(u_pred, u_valid.to(device))
                error_v_valid = Error().L2_error(v_pred, v_valid.to(device))
                error_p_valid = Error().L2_error(p_pred, p_valid.to(device))
                self.error_u.append(error_u_valid)
                self.error_v.append(error_v_valid)
                self.error_p.append(error_p_valid)
                # Save trained model (best performance)
                if (error_p_valid+error_u_valid+error_v_valid)/3. < best_err:
                    best_err = (error_p_valid + error_u_valid + error_v_valid)/3.
                    self._save(save_path, model_type='model_best_error')
                if (loss_train.item()) < best_loss:
                    best_loss = loss_train.item()
                    self._save(save_path, model_type='model_best_loss')
                # 
                if iter%100 == 0:
                    print(f"At iter: {iter+1}, loss_total:{np.mean(self.loss_list[-5:]):.3f}, error_u:{self.error_u[-1]:.3f}, error_v:{self.error_v[-1]:.3f}, error_p:{self.error_p[-1]:.3f}")
                    print('---loss:', torch.tensor(loss_all))
            # 
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            self.scheduler.step()
            iter += 1
        # Save trained model (final)
        self._save(save_path, model_type='model_final')
        print(f'The total training time is {time.time()-t_start:.4f}')