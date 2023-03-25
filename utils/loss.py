import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
relu=nn.ReLU()
def sdf_boundary(sdf,pts,threshold=0.8):
    norm_pts=pts.norm(dim=-1,keepdim=True)    
    #print(norm_pts.shape)
    mask=norm_pts>threshold
    loss_=relu(-sdf[mask]).sum()
    return loss_