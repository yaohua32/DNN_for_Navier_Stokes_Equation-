# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-21 20:26:35 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-21 20:26:35 
#  */
import numpy as np 
import torch
import sys

class Error():

    def __init__(self):
        pass 

    def L2_error(self, u_pred:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        '''
        err = torch.sum( (u_pred - u)**2 ) \
            / (torch.sum(u**2) + sys.float_info.epsilon)
        
        return torch.sqrt(err).item()