import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import open3d
import math


#assume all geometry lies in radius of 1;
#set the parameter as this then;
#somehow load a 
#there may be a large deformation...one may need to have vertex color instead;
class TriMesh(nn.Module):
    def __init__(self,radius=.7,tex_channel=3,resolution_mesh=4000,resolution_featmap=1024):
        super().__init__()
        sph_init=open3d.geometry.TriangleMesh.create_sphere(radius=radius,resolution=int(math.sqrt(resolution_mesh//4))+1,create_uv_map=True)
        vtx=torch.from_numpy(np.asarray(sph_init.vertices).astype(np.float32))
        tri=torch.from_numpy(np.asarray(sph_init.triangles).astype(np.int32))
        uv=torch.from_numpy(np.asarray(sph_init.triangle_uvs).astype(np.float32))
        uv_index=torch.from_numpy(np.arange(0,uv.shape[0]).astype(np.int32).reshape(uv.shape[0]//3,3))
        texmap=torch.zeros((resolution_featmap,resolution_featmap,tex_channel),dtype=torch.float32)
        vertex_color=torch.zeros((vtx.shape[0],tex_channel),dtype=torch.float32)
        
        #self.register_parameter("vertex",nn.Parameter(vtx))
        self.register_parameter("texmap",nn.Parameter(texmap))
        self.register_parameter("vertex_color",nn.Parameter(vertex_color))
        
        self.register_buffer("vertex",(vtx))
        self.register_buffer("uv",uv)
        self.register_buffer("uv_idx",uv_index)
        self.register_buffer("tri_idx",tri)

    def forward(self,tri_idx,barycentric):
        tri=self.tri_idx[tri_idx]
        color=self.vertex_color[tri[0]]*barycentric[0]+self.vertex_color[tri[1]]*barycentric[1]+self.vertex_color[tri[2]]*barycentric[2]
        return color

        

