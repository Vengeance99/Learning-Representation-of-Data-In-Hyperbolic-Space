
from turtle import forward
# from pyrsistent import T
import torch.nn.functional as F
import torch.nn as nn
import warnings   
import math
import Hypmath
import torch
from torch._C import _infer_size, _add_docstr

from torch import _VF

from torch.overrides import has_torch_function, handle_torch_function

import math


import torch
from torch import Tensor
Tensor.t
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init

class HypLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
   
    def forward(self, input: Tensor) -> Tensor:
        tens_ops = (input, self.weight)
        if not torch.jit.is_scripting():
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
                return handle_torch_function(forward, tens_ops, input, self.weight, bias=self.bias)

        if input.dim() == 2 and self.bias is not None:
            # print((self.weight).shape)
            # print(input.shape)
            tmp=Hypmath._mobius_matvec((self.weight),input,0.9)
            # print(tmp.shape)
            tmp=Hypmath._project(tmp,0.9)
            hypbias=Hypmath._expmap0(self.bias,0.9)
            hypbias=Hypmath._project(hypbias,0.9)
            # print((self.bias).shape)
            ret=Hypmath._mobius_add(tmp,hypbias,0.9)
            ret=Hypmath._project(ret,0.9)
            
        else:
            output = Hypmath._mobius_matvec((self.weight),input,0.9)
            output=Hypmath._project(output,0.9)
            if self.bias is not None:
                hypbias=Hypmath._expmap0(self.bias,0.9)
                hypbias=Hypmath._project(hypbias,0.9)
                output =Hypmath._mobius_add(output,hypbias,0.9)
                output=Hypmath._project(output,0.9)
            ret = output
        return ret
# m1 = HypLinear(28*28, 500)
# m2=HypLinear(500,250)
# m3=HypLinear(250,10)
# input = torch.randn(1,28*28)
# x = m1(input)
# x=m2(x)
# x=m3(x)
# print(x.size())        
# # n1=HypLinear(728,500)

class HypLinear1(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def _init_(self,in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self)._init_()
        # self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = Hypmath._mobius_matvec(drop_weight, x, self.c)
        res = Hypmath._project(mv, self.c)
        if self.use_bias: 
            bias = self.bias
            hyp_bias = Hypmath._expmap0(bias, self.c)
            hyp_bias = Hypmath._project(hyp_bias, self.c)
            res = Hypmath._mobius_add(res, hyp_bias, c=self.c)
            res = Hypmath._project(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
                self.in_features, self.out_features, self.c
        )