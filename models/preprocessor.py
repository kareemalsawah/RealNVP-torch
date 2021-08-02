import numpy as np
import torch
import torch.nn as nn

from ..utils import *

class Preprocessor(nn.Module):
    def __init__(self,max_val=1,alpha=0.05):
        super().__init__()
        self.max_val = max_val + 1
        self.alpha = alpha

    def forward(self,x,invert=False):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size,...)

        invert: bool
        
        Returns
        -------
        new_x: torch.FloatTensor, shape = x.shape
            Preprocessed x
        log_det_jac: torch.FloatTensor, shape = (batch_size,)
            Log determinant of the jacobian of the preprocessing transformations done on x
        '''
        if invert:
            x = torch.sigmoid(x)  # Inverse of logit
            new_x = (x-self.alpha)/(1-2*self.alpha)
            log_det_jac = torch.log(x) + torch.log(1-x) - np.log(1-2*self.alpha)

            return new_x, log_det_jac
        else:
            x = x + uniform_dist(0,1,x.shape)  # Dequantization

            # Logit Trick
            x /= self.max_val
            x = (1-2*self.alpha)*x + self.alpha
            new_x = torch.log(x) - torch.log(1-x)
            log_det_jac = np.log((1-2*self.alpha)/self.max_val) - torch.log(x) - torch.log(1-x) 

            return new_x, log_det_jac