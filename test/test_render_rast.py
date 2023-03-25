import torch
import numpy as np
import os

from src.camera_process.camera_process import KRt2GL
import nvdiffrast.torch as dr

import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="2"
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    #we may just interpolate the coordinates and get a otput;
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color,pos_clip,rast_out

cube_dir="./dummy_dataset"
fn="cube_c.npz"
with np.load(f'{cube_dir}/{fn}') as f:
    pos_idx, vtxp, col_idx, vtxc = f.values()

device="cuda"
f=64.
c=f//2
resolution=64

pos_idx=torch.from_numpy(pos_idx.astype(np.int32)).to(device)
vtxp=torch.from_numpy(vtxp).float().to(device)
col_idx=torch.from_numpy(col_idx.astype(np.int32)).to(device)
vtxc=torch.from_numpy(vtxc).float().to(device)
vtxc_exp=torch.cat([vtxc]*2,dim=-1)


K=torch.eye(3,dtype=torch.float32,device=device)
K[0,0]=f
K[1,1]=f
K[0,2]=c
K[1,2]=c

Rt=torch.zeros((3,4),dtype=torch.float32,device=device)
Rt[:3,:3]=torch.eye(3,dtype=torch.float32,device=device)
Rt[2,3]=2.

p,mv=KRt2GL(K,Rt,resolution,resolution,.3,3.)
mvp=torch.matmul(p,mv)

glctx = dr.RasterizeCudaContext()
color,pos_clip,rast_out=render(glctx,mvp,vtxp,pos_idx,vtxc_exp,col_idx,resolution)
print(color.shape)
img=(color[0,...,:3]*255.).cpu().numpy().astype(np.uint8)
cv2.imwrite("./test.png",img)

