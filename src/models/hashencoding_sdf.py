import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

import os

from src.modules.hashencoding_nv import SimpleHashEncoding
from src.modules.mlp_nv import SimpleMLP

class HashEncoding_SDF(nn.Module):
    def __init__(self,hidden_dim=128,num_layers=5,include_rbg=True):
        super().__init__()
        self.encoding=SimpleHashEncoding()
        self.include_rbg=include_rbg
        output_dim=4 if include_rbg else 1
        self.network=SimpleMLP(32,output_dim,hidden_dim,num_layers)
    def forward(self,x):
        enc=self.encoding(x)
        out=self.network(enc)
        return out
    def grad(self,x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf = self.forward(x)[...,-1]
            nablas = autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return nablas
        
