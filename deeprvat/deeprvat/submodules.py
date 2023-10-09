import copy
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from deeprvat.utils import pad_variants
     
            
class Layer_worker(pl.LightningModule):
    def __init__(self, activation, normalization, in_dim, out_dim, bias=True, solo=False):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias)
        self.solo = solo
        
        if not self.solo: 
            self.regularization = Regularization(out_dim, activation, normalization)
            if normalization == "spectral_norm":  
                self.layer = nn.utils.parametrizations.spectral_norm(self.layer)
    
    def forward(self, x):
        x = self.layer(x)
        if not self.solo: x = self.regularization(x)
        return x


class ResLayer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, layer, solo=False):
        super().__init__()
        self.layer_1 = layer.__getitem__(input_dim, input_dim)
        self.layer_2 = layer.__getitem__(input_dim, output_dim, solo=solo)
    
    def forward(self, x):
        out = self.layer_1(x) 
        out = self.layer_2(out) + x
        return  out


# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Bottleneck_ResLayer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, layer, solo=False):
        super().__init__()
        downsample_factor = 4
        sub_dim = input_dim // downsample_factor
        self.layer_1 = layer.__getitem__(input_dim, sub_dim)
        self.layer_2 = layer.__getitem__(sub_dim, sub_dim)
        self.layer_3 = layer.__getitem__(sub_dim, output_dim, solo=solo)
    
    def forward(self, x):
        out = self.layer_1(x)    
        out = self.layer_2(out)
        out = self.layer_3(out) + x
        return  out


class Layer(pl.LightningModule):
    def __init__(self, activation, normalization):
        super().__init__()   
        self.activation = activation
        self.normalization = normalization
    
    def __getitem__(self, in_dim, out_dim, bias=True, solo=False):
        return Layer_worker(self.activation, self.normalization, in_dim, out_dim, bias, solo)


class Regularization(pl.LightningModule):
    def __init__(self, in_dim, activation, normalization):
        super().__init__()
        self.activation = activation
        self.normalization = normalization
        self.do_normalization = True if self.normalization else False
        if self.normalization == "AnnotationNorm":
            self.normalization = Annotation_normalization(in_dim)
        else: self.do_normalization = False
    
    def forward(self, x):
        x = self.activation(x)
        if self.do_normalization: x = self.normalization(x)
        return x
        
class Annotation_normalization(pl.LightningModule):
    def __init__(self, in_dim, eps=0., momentum=0.99):
        super().__init__()
        self.eps = torch.tensor(eps, requires_grad=False) #1e-100
        self.momentum = torch.tensor(momentum, requires_grad=False)
        
        self.init = True
        self.mean = torch.nn.Parameter(torch.zeros((in_dim,), dtype=torch.float16))
        self.std = torch.nn.Parameter(torch.ones((in_dim,), dtype=torch.float16))
        self.mean.requires_grad = False
        self.std.requires_grad = False
    
    def reset_params(self, mean, std):
        self.mean += mean - self.mean
        self.std += std - self.std
    
    def update_params(self, x, mask):
        mean = torch.mean(x[mask], dim=0).detach()
        std = torch.std(x[mask], dim=0).detach()
        if self.init: 
            self.reset_params(mean, std)
            if x.is_cuda: 
                self.eps = self.eps.cuda()
                self.momentum = self.momentum.cuda()
            self.init = False
        else:
            if 0 < self.momentum < 1:
                self.mean *= self.momentum
                self.mean += mean * (1 - self.momentum)
                self.std *= self.momentum
                self.std += std * (1 - self.momentum)
            else: 
                self.reset_params(mean, std)
            
    def forward(self, x):
        # Exclude padded variants from mean and std computation and 
        # their application to the input tensor.
        # This applies only to inital layer, since all subsequent layers will have done
        # x' = (x * weight) + bias at least once, s. t. x' != 0.
        if x.ndim  == 4: mask = torch.where(x.sum(dim=3) == 0, False, True)
        else: mask = torch.ones(x.shape[:-1]).to(bool)
        if self.training: self.update_params(x, mask) 
        x[mask] = x[mask].sub(self.mean).div(self.std.add(self.eps)) 
        return x
    

class Layers(pl.LightningModule):
    def __init__(self, n_layers, bottleneck_layers, res_layers, input_dim, output_dim, internal_dim, activation, normalization, init_power_two, steady_dim, ):
        super().__init__()
        self.n_layers = n_layers
        self.bottleneck_layers = bottleneck_layers
        self.res_layers = res_layers
        self.layer = Layer(activation, normalization)
        
        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.output_dim = output_dim
        self.activation = activation
        self.init_power_two = init_power_two
        self.steady_dim = steady_dim
        
        self.layers, self.dims = self.get_architecture() 
        self.layer_dict = {0: {"layer": self.layer.__getitem__, "args": {}},
                           1: {"layer": ResLayer, "args": {"layer": self.layer}},
                           2: {"layer": Bottleneck_ResLayer, "args": {"layer": self.layer}}}

    def get_next_power_of_two(self, dim, factor):
        if factor == 2:
            return 2**list(filter(lambda x: (2**x > dim), range(10)))[0]
        else: return 2**list(filter(lambda x: (2**x <= dim), range(10)))[-1]
    
    def get_last_power_of_two(self, dim, factor):
        if factor == 2:
            return 2**list(filter(lambda x: (2**x < dim), reversed(range(10))))[0]
        else: return dim
    
    def get_operations(self):
        if self.input_dim < self.output_dim:
            operation, factor = min, 2
        elif self.input_dim > self.output_dim:
            operation, factor = max, 0.5
        else: 
            operation, factor = min, 1
        return operation, factor

    def get_ballon_dims(self):
        dims = []
        step_dim = self.input_dim
        if self.internal_dim: assert self.input_dim < self.internal_dim > self.output_dim
        for i in range(self.n_layers):
            input_dim = step_dim
            if self.steady_dim:
                buildup_layers = 1
                return_layer =  2
            else:
                to_power_two = list(filter(lambda x: (2**x >= self.input_dim), range(10)))[0]  
                to_dim = list(filter(lambda x: (2**x == self.internal_dim), reversed(range(10))))[0]
                if 2**to_power_two > self.input_dim: to_power_two -= 1 # none power two to power two
                buildup_layers = to_dim - to_power_two

                to_dim = list(filter(lambda x: (2**x == self.internal_dim), reversed(range(10))))[0]
                to_min = list(filter(lambda x: (2**x <= self.output_dim), reversed(range(10))))[0]
                return_layer = to_dim - to_min + 1 # internal_dim to min_dim, to 1
            
            intermediate_layers = self.n_layers - buildup_layers - return_layer - self.res_layers - self.bottleneck_layers
            if self.res_layers + self.bottleneck_layers <= i:
                if i < buildup_layers + self.bottleneck_layers + self.res_layers:
                    if self.steady_dim: step_dim = self.internal_dim
                    else: step_dim = min(self.internal_dim, self.get_next_power_of_two(step_dim, 2))
                elif i > intermediate_layers + buildup_layers + self.bottleneck_layers + self.res_layers:
                    if self.steady_dim and i == self.n_layers -1: step_dim = self.output_dim 
                    else: step_dim = max(self.output_dim, self.get_last_power_of_two(step_dim, 2))
            dims.append([int(input_dim), int(step_dim)]) 
        assert self.output_dim == step_dim
        return dims
    
    def get_dims(self):
        operation, factor = self.get_operations()
        dims = []
        step_dim = self.input_dim
        for i in range(self.n_layers):
            input_dim = step_dim
            if self.steady_dim: 
                if i == self.n_layers - 1: step_dim = self.output_dim    
            else:
                if i == 0 and self.init_power_two:
                    step_dim = operation(self.output_dim, self.get_next_power_of_two(input_dim, factor))
                else:
                    if self.res_layers + self.bottleneck_layers <= i:
                        if i == self.n_layers - 1: 
                            step_dim = self.output_dim 
                        else:
                            step_dim = operation(self.output_dim, self.get_next_power_of_two(input_dim, factor))
            dims.append([int(input_dim), int(step_dim)]) 
        assert self.output_dim == step_dim
        return dims
    
    def get_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0 and self.init_power_two:
                layers.append(0)
                self.res_layers += 1
            elif self.bottleneck_layers > i:
                layers.append(2)
            elif self.res_layers > i:
                layers.append(1)
            else:
                layers.append(0)
        return layers
    
    def get_architecture(self):
        assert not self.init_power_two or not self.steady_dim
        # assert self.n_layers > self.res_layers + self.bottleneck_layers
        layers = self.get_layers()
        if self.internal_dim: dims = self.get_ballon_dims()
        else: dims = self.get_dims()
        return layers, dims

    def get_layer(self, i, solo=False):
        layer = self.layer_dict[self.layers[i]]
        layer = layer["layer"](*self.dims[i], **layer["args"], solo=solo)
        return layer
    
    def get_layer_set_dim(self, i, in_dim, out_dim, bias=True, solo=False):
        layer = self.layer_dict[self.layers[i]]
        layer = layer["layer"](in_dim, out_dim, bias, **layer["args"], solo=solo)
        return layer

class WLC(nn.Module):
    def __init__(self, top=2, ranked=False):
        super().__init__()
        self.top = top
        # if ranked: self.factor = torch.tensor([1/n for n in range(1, top + 1)]).unsqueeze(1)
        # else: self.factor = torch.ones((1,1))
        self.factor = torch.ones((1,1))
        if torch.cuda.is_available(): self.factor = self.factor.cuda()

    def forward(self, x):
        values = torch.sort(x, dim=2, descending=True).values
        values = values[:, :, :self.top, :]
        values = torch.sum(values * self.factor, dim=2)
        return values
        
class Pooling(pl.LightningModule):
    def __init__(self, activation, normalization, pool, dim, n_variants):
        super().__init__()
        if pool not in ('sum', 'max','softmax', 'WLC'):  raise ValueError(f'Unknown pooling operation {pool}')
        self.layer = Layer(activation, normalization)
        self.pool = pool
        self.dim = dim
        self.n_variants = n_variants

        self.f, self.f_args = self.get_function()
            
    def get_function(self):
        if self.pool == "sum": return torch.sum, {"dim": 2} 
        elif self.pool == 'softmax':
            '''
                Modeled after Enformer from DeepMind
                paper: https://www.nature.com/articles/s41592-021-01252-x 
                original code: https://github.com/deepmind/deepmind-research/blob/cb555c241b20c661a3e46e5d1eb722a0a8b0e8f4/enformer/enformer.py#L244  
                pytorch remade code: https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py#L134 
            '''
            self.to_attn_logits = self.layer.__getitem__(self.n_variants, self.n_variants, False) #bias = False
            nn.init.eye_(self.to_attn_logits.layer.weight)
            self.gain = 2.0 #When 0.0 is equivalent to avg pooling, and when ~2.0 and `per_channel=False` it's equivalent to max pooling.
            with torch.no_grad(): self.to_attn_logits.layer.weight.mul_(self.gain)
            return torch.sum, {"dim": -1}  
        elif self.pool == 'WLC':
            wlc = WLC()
            return wlc.forward, {}
        else: return torch.max, {"dim": 2} 

    def forward(self, x):
        if self.pool == "softmax":
            x = x.permute((0,1,3,2))
            if x.shape[-1] < self.n_variants: x = pad_variants(x,self.n_variants)  
            x = x.unsqueeze(3)  #Rearrange('b g (v p) l -> b g l v p', p = self.pool_size)
            x = x * self.to_attn_logits(x).softmax(dim=-1)

        x = self.f(x, **self.f_args)

        if self.pool == "softmax": x = x.squeeze(-1)
        if self.pool == "max": x = x.values
        return x

NORMALIZATION = {"SpectralNorm":  nn.utils.parametrizations.spectral_norm,
                 "LayerNorm":  nn.LayerNorm,
                 "BatchNorm": nn.BatchNorm1d,
                 "AnnotationNorm": Annotation_normalization}
