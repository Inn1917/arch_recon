import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extern.diffusion_model.sd import StableDiffusion

from src.models.neus_sdf_normal import SDFNetwork,RenderingNetwork,NeRF,SingleVarianceNetwork
from src.renderer.neus_renderer_normal import NeuSRenderer

from src.dataset.rand_cam import CamDataset
import random

from extern.nvdiffrec_core.geometry.dmtet import DMTetGeometry,sdf_reg_loss
import nvdiffrast.torch as dr
from extern.nvdiffrec_core.render.obj import write_obj

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
#def get_geo_loss(vtx,tri_idx):
#something else;
#about the mesh rendering business;
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, vtx_nml, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    pos_map, _    = dr.interpolate(pos[None, ...], rast_out, pos_idx)
    #we may just interpolate the coordinates and get a otput;
    pos_map       = dr.antialias(pos_map, rast_out, pos_clip, pos_idx)

    normal_map, _    = dr.interpolate(vtx_nml[None, ...], rast_out, pos_idx)
    #we may just interpolate the coordinates and get a otput;
    normal_map       = dr.antialias(normal_map, rast_out, pos_clip, pos_idx)
    return pos_map,normal_map
def rays_d_from_KRt(K,Rt,resolution):
    device=K.device
    pix=torch.ones((resolution,resolution,3),dtype=torch.float32,device=device)
    x=torch.arange(0,resolution,step=1,dtype=torch.float32,device=device)
    y=torch.arange(0,resolution,step=1,dtype=torch.float32,device=device)
    pix[:,:,0]=x[None,:]
    pix[:,:,1]=y[:,None]

    K_inv=K.inverse()
    rays_d_pix=torch.matmul(K_inv[None,None,:,:],pix[:,:,:,None])
    rays_d=torch.matmul(Rt[:3,:3].transpose(1,0)[None,None,:,:],rays_d_pix)

    return rays_d
if __name__=="__main__":
    import os
    import cv2
    from pyhocon import ConfigFactory
    import trimesh
    from src.camera_process.camera_process import KRt2GL

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device="cuda"
    seed_everything(0)
    parser = argparse.ArgumentParser(description='mesh_diffusion_model')
    parser.add_argument('--radius', type=float, default=.7, help='the initial radius of mesh')
    parser.add_argument('--tex_channel', type=int, default=3, help='the tex map channel')
    parser.add_argument('--resolution_mesh', type=int, default=2500)
    parser.add_argument('--resolution_featmap', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    opt = parser.parse_args()

    conf_path = "./confs/womask_normal.conf"
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    #self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)


    sd_model=StableDiffusion(device)
    text_ebd=sd_model.get_text_embeds(["a high quality photo of a pineapple"],[""])
    sdf_network=SDFNetwork(**conf['model.sdf_network']).to(device)
    color_network=RenderingNetwork(**conf['model.rendering_network']).to(device)
    sdf_network.load_state_dict(torch.load("./saved_model/27000_sdf.pth"))
    color_network.load_state_dict(torch.load("./saved_model/27000_color.pth"))
    #change to 4. for proper range;
    geometry=DMTetGeometry(128,2.4,opt,sdf_network).to(device)
    
    params_to_train=[]
    params_to_train += list(geometry.parameters())
    params_to_train += list(color_network.parameters())
    params_to_train += list(sdf_network.parameters())
    
    #del sdf_network
    optimizer=torch.optim.Adam(params_to_train,lr=5e-4)
    glctx = dr.RasterizeCudaContext()
    dataset=CamDataset(cam_size=30)
    resolution=512
    for i in range(opt.max_iter):
        K,Rt=dataset[0]
        K=K.to(device)
        Rt=Rt.to(device)
        K[:2]*=resolution//64

        p,mv=KRt2GL(K,Rt,resolution,resolution,.8,4.5)
        mvp=torch.matmul(p,mv)

        mesh=geometry.getMesh(material=None)
        v,t=mesh.v_pos,mesh.t_pos_idx
        t=t.int()
        mesh_py3d=Meshes(verts=[v],faces=[t])
        loss_laplace=mesh_laplacian_smoothing(mesh_py3d)

        n=mesh.v_nrm
        pts_img,nml_img=render(glctx,mvp,v,t,n,resolution)
        
        pts_in=pts_img.reshape(-1,3)
        rays_d=rays_d_from_KRt(K,Rt,resolution).reshape(-1,3)
        rays_d=rays_d/rays_d.norm(dim=-1,keepdim=True)
        normal=nml_img.reshape(-1,3)
        normal=(normal/(normal.norm(dim=-1,keepdim=True)+1e-10)).squeeze()
        
        feat_vec=sdf_network.forward(pts_in)[:,1:]
        color=color_network.forward(pts_in,normal,rays_d,feat_vec).reshape(1,resolution,resolution,3)
        #print(color.shape)
        color_=color.permute(0,3,1,2)
        loss_sdf=sdf_reg_loss(geometry.sdf,geometry.all_edges)
        loss_sds=sd_model.train_step_full_res(text_ebd,color_)
        loss=loss_sds+5e-4*loss_sdf+5e-3*loss_laplace
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            print(i,": ",loss.data,loss_laplace.data)
        if i%500==0:

            img=color[0]
            img=(img*255.).detach().cpu().numpy().astype(np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./result_refine/img/{i:03d}.png",img)
            write_obj("./result_refine",mesh)
            print("saved image")
            