import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.mesh import TriMesh
from src.camera_process.camera_process import KRt2GL
import nvdiffrast.torch as dr
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color,pos_clip,rast_out

class TriMesh_Renderer(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.mesh=TriMesh(
            opt.radius,
            opt.tex_channel,
            opt.resolution_mesh,
            opt.resolution_featmap
        )
        self.resolution=opt.resolution#int square;
    def forward(self,K,Rt,resolution=None):
        if resolution is None:
            resolution=self.resolution
        #alright;
        norm=Rt[:,3].norm()
        p,mv=KRt2GL(K,Rt,resolution,resolution,norm-1.2,norm+1.2)
        mvp=torch.matmul(p,mv)
        glctx=dr.RasterizeCudaContext()
        color,_,_=render(glctx,mvp,self.mesh.vertex,self.mesh.tri_idx,torch.sigmoid(self.mesh.vertex_color),self.mesh.tri_idx,resolution)
        return color
