import copy
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from deeprvat.utils import pad_variants
     
            
class Layer_worker(nn.Module):
    def __init__(self, activation, normalization, in_dim, out_dim, bias=True, solo=False):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias)
        self.normalization = normalization
        self.solo = solo

        if not self.solo: 
            if self.normalization == "spectral_norm":  
                self.layer = nn.utils.parametrizations.spectral_norm(self.layer)
            self.regularization = Regularization(activation, normalization)

    def forward(self, x):
        x = self.layer(x)
        if not self.solo: x = self.regularization(x)
        return x
            

class Layer(nn.Module):
    def __init__(self, activation, normalization):
        super().__init__()   
        self.activation = activation
        self.normalization = normalization
    
    def __getitem__(self, in_dim, out_dim, bias=True, solo=False):
        return Layer_worker(self.activation, self.normalization, in_dim, out_dim, bias, solo)
    
    
class Regularization(nn.Module):
    def __init__(self, activation, normalization):
        super().__init__()
        self.activation = activation
        self.normalization = normalization
        self.do_activation = True if self.activation else False
        self.do_normalization = True if self.normalization else False
        
        if self.normalization == "AnnotationNorm":
            self.normalization = Annotation_Normalization()
        else: self.do_normalization = False
    
    def switch(self):
        self.do_activation = not self.do_activation
        self.do_normalization = not self.do_normalization
    
    def forward(self, x):
        if self.do_activation: x = self.activation(x)
        if self.do_normalization: x = self.normalization(x)
        return x

class ResLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer, solo=False):
        super().__init__()
        self.layer_1 = layer.__getitem__(input_dim, input_dim)
        self.layer_2 = layer.__getitem__(input_dim, output_dim, solo=solo)
    
    def switch(self):
        self.layer_2.switch()
    
    def forward(self, x):
        out = self.layer_1(x) 
        out = self.layer_2(out) + x
        return  out

# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Bottleneck_ResLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer, solo=False):
        super().__init__()
        downsample_factor = 4
        sub_dim = input_dim // downsample_factor
        self.layer_1 = layer.__getitem__(input_dim, sub_dim)
        self.layer_2 = layer.__getitem__(sub_dim, sub_dim)
        self.layer_3 = layer.__getitem__(sub_dim, output_dim, solo=solo)
    
    def switch(self):
        self.layer_3.switch()
    
    def forward(self, x):
        out = self.layer_1(x)    
        out = self.layer_2(out)
        out = self.layer_3(out) + x
        return  out

class Annotation_Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.tensor(1e-100, requires_grad=False)
        # self.momentum = torch.tensor(0.99, requires_grad=False)
        self.momentum = torch.tensor(0., requires_grad=False)
        self.init = True
        
    def forward(self, x):
        if x.ndim  == 4:
            # exclude padded variants from mean, std computation and standardization 
            # application to tensor
            mask = torch.where(x.sum(dim=3) == 0, False, True)
        else: mask = torch.ones(x.shape[:-1]).to(bool)
        if x.ndim == 2: dims = 0
        else: dims = list(range(0, x.ndim - 1))
        
        mean = torch.mean(x[mask], dim=0).detach()
        std = torch.std(x[mask], dim=0).detach()

        if self.init: 
            if x.is_cuda: self.momentum = self.momentum.cuda()
            self.mean = mean
            self.std = std
            self.init = False
        else:
            if self.momentum > 0:
                self.mean = self.mean * self.momentum + mean * (1 - self.momentum)
                self.std = self.std * self.momentum + std * (1 - self.momentum)
            else: 
                self.mean = mean
                self.std = std
        
            x[mask] = x[mask].sub(self.mean).div(self.std.add(self.eps)) 
        return x

class Layers(nn.Module):
    def __init__(self, n_layers, bottleneck_layers, res_layers, input_dim, output_dim, activation, normalization, init_power_two, steady_dim):
        super().__init__()
        self.n_layers = n_layers
        self.bottleneck_layers = bottleneck_layers
        self.res_layers = res_layers
        self.layer = Layer(activation, normalization)
        
        self.input_dim = input_dim
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
            return 2**list(filter(lambda x: (2**x >= dim), range(10)))[0]
        else:
            return 2**list(filter(lambda x: (2**x <= dim), range(10)))[-1]
    
    def get_operations(self):
        if self.input_dim < self.output_dim:
            operation, factor = min, 2
        elif self.input_dim > self.output_dim:
            operation, factor = max, 0.5
        else: 
            operation, factor = min, 1
        return operation, factor

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
                            if input_dim not in [2**i for i in range(10)]:
                                step_dim = operation(self.output_dim, self.get_next_power_of_two(input_dim, factor))
                            else: 
                                step_dim = operation(self.output_dim, input_dim * factor)
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
        dims = self.get_dims()
        return layers, dims

    def get_layer(self, i, solo=False):
        layer = self.layer_dict[self.layers[i]]
        layer = layer["layer"](*self.dims[i], **layer["args"], solo=solo)
        return layer
    
    def get_layer_set_dim(self, i, in_dim, out_dim, bias=True, solo=False):
        layer = self.layer_dict[self.layers[i]]
        layer = layer["layer"](in_dim, out_dim, bias, **layer["args"], solo=solo)
        return layer

class Pooling(pl.LightningModule):
    def __init__(self, activation, normalization, pool, dim, n_variants):
        super().__init__()
        if pool not in ('sum', 'max','softmax'):  raise ValueError(f'Unknown pooling operation {pool}')
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
                 "AnnotationNorm": Annotation_Normalization}