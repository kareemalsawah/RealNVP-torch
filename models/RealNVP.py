import numpy as np
import torch
import torch.nn as nn

from .preprocessor import *
from ..utils import *

class ResBlock(nn.Module):
    def __init__(self,n_filters):
        super().__init__()

        self.convs = nn.Sequential(nn.Conv2d(n_filters,n_filters,kernel_size=(1,1)),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(n_filters,n_filters,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(n_filters,n_filters,kernel_size=(1,1)))

    def forward(self,x):
        h = self.convs(x)
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
        #self.conv_2.weight.data *= 0
        #self.conv_2.bias.data *= 0

    def forward(self,x):
        x = self.conv_1(x)
        x = self.layers(x)
        x = self.conv_2(x)

        return x

class ActNorm(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.C = C

        self.log_scale = nn.Parameter(torch.zeros((1,self.C,1,1)))
        self.shift = nn.Parameter(torch.zeros((1,self.C,1,1)))

        self.first_batch = True

    def forward(self,x,invert=False):
        if invert:
            assert not self.first_batch
            return (x - self.shift) * torch.exp(-1*self.log_scale), -1*self.log_scale*torch.ones(x.shape).to(self.shift.device)
        else:
            if self.first_batch:
                self.log_scale.data = -1*safe_log(torch.std(x,dim=(0,2,3))+1e-4).reshape(1,self.C,1,1)
                self.shift.data = -1*torch.mean(x,dim=(0,2,3)).reshape(1,self.C,1,1)

                self.first_batch = False
            return x * torch.exp(self.log_scale) + self.shift, self.log_scale*torch.ones(x.shape).to(self.shift.device)

class one_one_conv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        pass

class AffineCoupling(nn.Module):
    order_type = 0
    def __init__(self,in_shape,n_filters=128,use_bn=False):
        AffineCoupling.order_type += 1
        super().__init__()

        self.mask = None
        self.use_bn = use_bn
        self.net = SimpleResnet(in_channels=in_shape[0],n_out=2,n_filters=n_filters)
        self.scale = nn.Parameter(torch.zeros(1))
        self.scale_shift = nn.Parameter(torch.zeros(1))

    def forward(self,x,invert=False):
        assert self.mask is not None

        x_ = x*self.mask
        log_scale, t = torch.chunk(self.net(x_), 2, dim=1)
        log_scale = self.scale*torch.tanh(log_scale) + self.scale_shift

        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        if invert:
            z = (x - t) / (torch.exp(log_scale) + 1e-4)
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
        # x = np.arange(8*8).reshape(8,8)
        # print(x)
        # x.reshape(8,2,2,2).transpose(0,3,1,2)
        for h in range(H):
            counter = h%2
            for w in range(W):
                if counter%2 == 0:
                    self.mask[0][h][w] = 0
                counter += 1

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

        self.mask = nn.Parameter(self.mask)
        self.mask.requires_grad = False

class AffineCouplingChannel(AffineCoupling):
    def __init__(self,in_shape):
        super().__init__(in_shape)
        C,H,W = in_shape

        self.mask = torch.ones(C,1,1)

        self.mask[:C//2] *= 0

        if AffineCoupling.order_type%2 == 0:
            self.mask = 1 - self.mask

        self.mask = nn.Parameter(self.mask)
        self.mask.requires_grad = False

class FlowSequential(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x, invert=False):
        log_det_jac = None

        layers = self.layers
        if invert:
            layers = layers[::-1]

        for layer in layers:
            x, log_det = layer.forward(x,invert=invert)
            if log_det_jac is None:
                log_det_jac = log_det
            else:
                log_det_jac += log_det

        return x, log_det_jac

def squeeze(x):
    bs,c,sl,_ = x.shape
    assert c==1

    squeezed = x.reshape(bs,sl//2,2,sl//2,2).permute(0,1,3,2,4).reshape(bs,sl//2,sl//2,4).permute(0,3,1,2)
    return squeezed

def unsqueeze(x):
    bs,c,sl,_ = x.shape
    assert c==4

    unsqueezed = x.permute(0,2,3,1).reshape(bs,sl,sl,2,2).permute(0,1,3,2,4).reshape(bs,1,2*sl,2*sl)
    return unsqueezed

class RealNVP(nn.Module):
    def __init__(self,in_shape,z_dist,max_val=1,large_model=False):
        super().__init__()
        self.in_shape = in_shape
        self.z_dist = z_dist
        self.is_z_simple = isinstance(self.z_dist, torch.distributions.Distribution)
        self.preprocess = Preprocessor(max_val)
        self.large_model = large_model

        self.layers_1 = FlowSequential([AffineCouplingCheckboard(in_shape),
                                        ActNorm(1),
                                        AffineCouplingCheckboard(in_shape),
                                        ActNorm(1),
                                        AffineCouplingCheckboard(in_shape)])
        scale_2_shape = (in_shape[0]*4,in_shape[1]//2,in_shape[2]//2)
        self.layers_2 = FlowSequential([AffineCouplingChannel(scale_2_shape),
                                        ActNorm(4),
                                        AffineCouplingChannel(scale_2_shape),
                                        ActNorm(4),
                                        AffineCouplingChannel(scale_2_shape),
                                        ActNorm(4)])
        if self.large_model:
            self.layers_3 = FlowSequential([AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape)])
            self.layers_4 = FlowSequential([AffineCouplingChannel(scale_2_shape),
                                            ActNorm(4),
                                            AffineCouplingChannel(scale_2_shape),
                                            ActNorm(4),
                                            AffineCouplingChannel(scale_2_shape),
                                            ActNorm(4)])
            self.layers_5 = FlowSequential([AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape)])
        else:
            self.layers_3 = FlowSequential([AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape),
                                            ActNorm(1),
                                            AffineCouplingCheckboard(in_shape)])

    def forward(self,x,invert=False):
        if invert:
            if self.large_model:
                x, log_det_5 = self.layers_5.forward(x,invert)
                x = squeeze(x)
                x, log_det_4 = self.layers_4.forward(x,invert)
                x = unsqueeze(x)
            x, log_det_3 = self.layers_3.forward(x,invert)
            x = squeeze(x)
            x, log_det_2 = self.layers_2.forward(x,invert)
            x = unsqueeze(x)
            x, log_det_1 = self.layers_1.forward(x,invert)
            x, log_det_pre = self.preprocess.forward(x,invert)
        else:
            x, log_det_pre = self.preprocess.forward(x,invert)
            x, log_det_1 = self.layers_1.forward(x,invert)
            x = squeeze(x)
            x, log_det_2 = self.layers_2.forward(x,invert)
            x = unsqueeze(x)
            x, log_det_3 = self.layers_3.forward(x,invert)
            if self.large_model:
                x = squeeze(x)
                x, log_det_4 = self.layers_4.forward(x,invert)
                x = unsqueeze(x)
                x, log_det_5 = self.layers_5.forward(x,invert)

        # Sum up log determinants
        log_det_jac = torch.sum(log_det_pre,dim=(1,2,3))
        log_det_jac += torch.sum(log_det_1,dim=(1,2,3))
        log_det_jac += torch.sum(log_det_2,dim=(1,2,3))
        log_det_jac += torch.sum(log_det_3,dim=(1,2,3))
        if self.large_model:
            log_det_jac += torch.sum(log_det_4,dim=(1,2,3))
            log_det_jac += torch.sum(log_det_5,dim=(1,2,3))
        return x, log_det_jac

    def log_prob(self, x, invert=False):
        num_dims = self.in_shape[0]*self.in_shape[1]*self.in_shape[2]
        if self.is_z_simple:
            assert invert == False

        x, log_det_jac = self.forward(x, invert=invert)
        if self.is_z_simple:
            log_prob_z = self.z_dist.log_prob(x)
            log_prob_z = torch.sum(log_prob_z,dim=(1,2,3))
            log_prob_x = log_prob_z + log_det_jac
            return log_prob_x/num_dims

        if invert:
            log_prob_x, log_det = self.z_dist.log_prob(x, invert=True)
            log_prob_z = log_prob_x + log_det + log_det_jac
            return log_prob_z/num_dims
        else:
            log_prob_z, log_det = self.z_dist.log_prob(x, invert=False)
            log_prob_x = log_prob_z + log_det + log_det_jac
            return log_prob_x/num_dims
