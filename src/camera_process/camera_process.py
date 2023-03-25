import torch
import numpy as np

def KRt2GL(K,Rt,width,height,near,far):
    '''
    [2*K00/width,  -2*K01/width,   (width - 2*K02 + 2*x0)/width,                            0]
    [          0, -2*K11/height, (height - 2*K12 + 2*y0)/height,                            0]
    [          0,             0, (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)]
    [          0,             0,                             -1,                            0]
    '''
    device=K.device
    mv=torch.eye(4,dtype=torch.float32,device=device)
    mv[:3]=Rt
    #mv[0,:]*=-1.
    mv[1:3,:]*=-1.
    p=torch.zeros((4,4),dtype=torch.float32,device=device)
    p[0,0]=2.*K[0,0]/width
    p[0,1]=-2.*K[0,1]/width
    p[0,2]=(width-2.*K[0,2])/width
    p[1,1]=-2.*K[1,1]/height
    p[1,2]=(height - 2*K[1,2])/height
    p[2,2]=(-near-far)/(far-near)
    p[2,3]=-2.*far*near/(far-near)
    p[3,2]=-1
    return p,mv