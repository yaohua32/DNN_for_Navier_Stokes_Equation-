# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:27:31 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:27:31 
#  */
import numpy as np
import torch

class Example():

    def __init__(self, np_type=np.float32, torch_type=torch.float32):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type

    def NS_Solver(self, save_path:str, load_path:str, load_type:str,
                  action:str, t0_loc:int=0, tT_loc:int=101, **args):
        '''
        '''
        from Solvers.PDE_Solver import PDE_Solver
        ############
        kwargs = {'loss_type': 'weak',
                  'Num_particles': 6,
                  'Num_tin_size': 600,
                  'Nx_integral': 10,
                  'train_xbd_size_each_face': 6,
                  'train_tbd_size': 600,
                  'maxIter': 200,
                  'lr': args['lr'],
                  'lrDecay': 1.,
                  'weight':{'eq':5., 'bd':2., 'u':5., 'v':5., 'p':10.},
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'net_type': 'tanh',
                  'test_fun': 'Wendland',
                  'stretch': 1.,
                  'int_method': 'mesh',
                  'hidden_width': 200,
                  'hidden_layer': 2,
                  'dtype':{'numpy':self.np_type, 'torch':self.torch_type}
                  }
        ##############
        problem = Problem(dtype=self.np_type, t0_loc=t0_loc, tT_loc=tT_loc)
        solver = PDE_Solver(problem=problem, **kwargs)
        if action=='train':
            solver.train(save_path, load_path, model_type=load_type)
        else:
            return solver.test(load_path, model_type=load_type)

    def train(self, save_path:str, load_path:str, tloc_list:list):
        '''
        '''
        lr_list = [1e-3, 1e-3, 0.5*1e-4, 0.5*1e-4]
        for inx in range(len(lr_list)):
            for tloc in tloc_list:                
                demo.NS_Solver(save_path=save_path+f't{tloc[0]}_{tloc[1]}/', 
                            load_path=load_path+f't{tloc[0]}_{tloc[1]}/',
                            load_type='model_best_error', action='train',
                            t0_loc=tloc[0], tT_loc=tloc[1], **{'lr':lr_list[inx]})

if __name__=='__main__':
    from Problems.NS_unsteady import Problem
    demo = Example(np_type=np.float32, torch_type=torch.float32)
    tloc_list = [(0,101)]
    ###########################################
    path = f"./code/savedModel/NS_unsteady/"
    demo.train(save_path=path, load_path=path, tloc_list=tloc_list)
