import numpy as np
import torch
import torch.nn as nn

from .preprocessor import *
from ..utils import *

class ResBlock(nn.Module):
    def __init__(self,n_filters):
        super().__init__()

        self.conv_1 = nn.Conv2d(n_filters,n_filters,kernel_size=(1,1))
        self.conv_2 = nn.Conv2d(n_filters,n_filters,kernel_size=(3,3),padding=1)
        self.conv_3 = nn.Conv2d(n_filters,n_filters,kernel_size=(1,1))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        h = self.relu(self.conv_1(x))
        h = self.relu(self.conv_2(h))
        h = self.conv_3(h)

        return h + x

class SimpleResnet(nn.Module):
    def __init__(self,in_channels,n_out,n_filters=128,n_blocks=8):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels,n_filters,kernel_size=(3,3),padding=1)
        layers = []
        for i in range(n_blocks):
            layers.append(ResBlock(n_filters))
        
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        
        self.conv_2 = nn.Conv2d(n_filters,n_out,kernel_size=(3,3),padding=1)
    
    def forward(self,x):
        x = self.conv_1(x)
        x = self.layers(x)
        x = self.conv_2(x)

        return x

class AffineCoupling(nn.Module):
    order_type = 0
    def __init__(self,in_shape,use_bn=False):
        AffineCoupling.order_type += 1
        super().__init__()

        self.mask = None
        self.use_bn = use_bn
        self.net = SimpleResnet(in_shape[0],2)
    
    def forward(self,x,invert=False):
        assert self.mask is not None

        x_ = x*self.mask
        log_scale, t = torch.chunk(self.net(x_), 2, dim=1)
        
        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        if invert:
            z = (x - t) / torch.exp(log_scale)
            log_det_jacobian = -1*log_scale
        else:
            z = x * torch.exp(log_scale) + t
            log_det_jacobian = log_scale

        return z, log_det_jacobian

class AffineCouplingCheckboard(AffineCoupling):
    def __init__(self,in_shape):
        '''
        Parameters
        ----------
        in_shape: tuple[int]
            Shape of the inputs represented as (C,H,W)
        '''
        super().__init__(in_shape)
        C,H,W = in_shape

        self.mask = torch.ones(1,H,W)

        # Inefficient way to do a checkboard pattern, but it shouldn't matter as it is only in the initialization
        for h in range(H):
            counter = h%2
            for w in range(W):
                if counter%2 == 0:
                    self.mask[0][h][w] = 0
                counter += 1

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

class AffineCouplingChannel(AffineCoupling):
    def __init__(self,in_shape):
        super().__init__(in_shape)
        C,H,W = in_shape

        self.mask = torch.ones(C,1,1)

        self.mask[:C//2] *= 0

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

class RealNVP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.preprocess = Preprocessor()

        # self.layers_1 = 

        # self.layers_2 = 
        
        # self.layers_3 = 
    
    def forward(self,x,invert=False):

        if invert:
            x,log_det_1 = self.layers_3.forward(x,invert)
            # Squeeze
            x,log_det_2 = self.layers_2.forward(x,invert)
            # Unsqueeze
            x,log_det_3 = self.layers_1.forward(x,invert)
            x, log_det_pre = self.preprocess.forward(x,invert)
        else:
            x, log_det_pre = self.preprocess.forward(x,invert)
            x,log_det_1 = self.layers_1.forward(x,invert)
            # Squeeze
            x,log_det_2 = self.layers_2.forward(x,invert)
            # Unsqueeze
            x,log_det_3 = self.layers_3.forward(x,invert)
        
        # Sum up log determinants
        # log_det = 
        return x, log_det