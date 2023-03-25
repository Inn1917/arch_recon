import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.modules.mlp_nv import SimpleMLP as MLP
from src.modules.neus_embedder import get_embedder
#noisy.
#get some layer norm?
class Triplane_NeRF(nn.Module):
    def __init__(self,feat_res=512,feat_dim=48,mlp_layer_num=4,mlp_layer_width=128):
        super().__init__()
        #the triplane sdf within +-1 cube;
        feat=torch.randn(3,feat_dim,feat_res,feat_res)
        self.register_parameter("feat",nn.Parameter(feat))
    
        self.ebd,self.ebd_dim=get_embedder(multires=4,input_dims=3)
        self.ebd_x,self.ebd_dim_x=get_embedder(multires=10,input_dims=3)
        self.mlp=MLP(feat_dim+self.ebd_dim+self.ebd_dim_x,4,mlp_layer_width,mlp_layer_num)
        self.activation = torch.nn.ReLU()#nn.Softplus(beta=10)
        
    def forward(self,x,d):
        #x:[r,d,3] sampling coord;
        x=x/3.
        d_=d[:,None,:].repeat(1,x.shape[1],1)
        xy=x[None,...,:2]
        yz=x[None,...,1:]
        xz=torch.cat([x[None,...,:1],x[None,...,2:]],dim=-1)
        #print(x.max(),x.min())
        feat_xy=F.grid_sample(self.feat[:1,...],xy,align_corners=True)
        feat_yz=F.grid_sample(self.feat[1:2,...],yz,align_corners=True)
        feat_xz=F.grid_sample(self.feat[2:,...],xz,align_corners=True)
        #how it is improved?
        feat_sum=(feat_xy+feat_yz+feat_xz)
        out=self.mlp.forward(torch.cat([(feat_sum[0,...].permute(1,2,0)),self.ebd(d_),self.ebd_x(x)],dim=-1))
        #out=self.mlp.forward(x)
        #(torch.sigmoid(out[...,:3])*(1.+2*0.001)-0.001)
        c,sigma=((out[...,:3])),self.activation(out[...,3:])
        
        return torch.cat([c,sigma],dim=-1)


