import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.mask = None
        self.net = SimpleResnet(in_channels,out_channels)
    
    def forward(self,x,invert=False):
        assert self.mask is not None

        x_ = x*self.mask
        log_s, t = torch.chunk(self.net(x_), 2, dim=1)
        
        t = t * (1.0 - mask)
        log_scale = log_scale * (1.0 - mask)
        z = x * torch.exp(log_scale) + t
        log_det_jacobian = log_scale
        return z, log_det_jacobian

class AffineCouplingCheckboard(AffineCoupling):
    order_type = 0
    def __init__(self,in_channels,out_channels):
        super().__init__(in_channels,out_channels)
        AffineCouplingCheckboard.order_type += 1

        if self.order_type%2 == 0:
            pass
        else:
            pass

class AffineCouplingChannel(AffineCoupling):
    order_type = 0
    def __init__(self,in_channels,out_channels):
        super().__init__(in_channels,out_channels)
        AffineCouplingChannel.order_type += 1

        if self.order_type%2 == 0:
            pass
        else:
            pass

class ActNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        pass

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