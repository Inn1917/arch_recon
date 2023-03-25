import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extern.diffusion_model.sd import StableDiffusion

from src.models.neus_sdf_normal import SDFNetwork,RenderingNetwork,NeRF,SingleVarianceNetwork,RenderingNetwork_NF
from src.renderer.nerf_renderer import NeRFRenderer

from src.dataset.rand_cam import CamDataset
from utils.loss import sdf_boundary
import random
#def get_geo_loss(vtx,tri_idx):
#something else;
#about the mesh rendering business;
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

PI=3.1415926535
if __name__=="__main__":
    import os
    import cv2
    from pyhocon import ConfigFactory
    import trimesh

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device="cuda"
    seed_everything(1919)
    parser = argparse.ArgumentParser(description='mesh_diffusion_model')
    parser.add_argument('--radius', type=float, default=.7, help='the initial radius of mesh')
    parser.add_argument('--tex_channel', type=int, default=3, help='the tex map channel')
    parser.add_argument('--resolution_mesh', type=int, default=2500)
    parser.add_argument('--resolution_featmap', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--max_iter', type=int, default=30000)
    opt = parser.parse_args()

    conf_path = "./confs/womask_nerf.conf"
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    #print(conf["model.sdf_network.scale"])
    #conf["model.sdf_network.scale"]=1.5
    #self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)

    sdf_network=SDFNetwork(**conf['model.sdf_network']).to(device)
    color_network=RenderingNetwork(**conf['model.rendering_network']).to(device)
    variance_network=SingleVarianceNetwork(**conf['model.variance_network']).to(device)
    nerf_outside = None
    nerf_inside=NeRF(**conf['model.nerf']).to(device)
    #sdf_network.load_state_dict(torch.load("./saved_model/27000_sdf.pth"))
    #color_network.load_state_dict(torch.load("./saved_model/27000_color.pth"))
    renderer=NeRFRenderer(nerf_outside,
                        nerf_inside,
                        **conf['model.neus_renderer'])

    
    sd_model=StableDiffusion(device,sd_version="1.5")
    params_to_train=[]
    params_to_train += list(nerf_inside.parameters())
    optimizer=torch.optim.Adam(params_to_train,lr=2.5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.7)

    #and then what? do some remeshing by the tetrahedral thing?
    #first thing first should deal with the "not in the middle" error;
    #still needs to be treated as liquid mesh;
    #first thing first try some text prompt; whether the "in the middle of the image" works on this;
    #then try some more views over a larger random range;
    
    dataset=CamDataset(cam_size=100)
    L=len(dataset)
    for i in range(opt.max_iter):
        K,Rt,theta,phi=dataset[i%L]
        K=K.to(device)
        Rt=Rt.to(device)
        if theta<(1+1/4.)*PI and theta>(1-1/4.)*PI:
            direction="front view"
        elif (theta<(1-1/4.)*PI and theta>(1./4)*PI) or (theta>(1+1/4.)*PI and theta<(2.-1./4)*PI):
            direction="side view"
        else:
            direction="back view"
        if phi>PI/3.:
            direction="overhead view"

        text_ebd=sd_model.get_text_embeds(["a high quality photo of a pineapple, "+direction],[""])
        color=renderer.forward(K,Rt)
        #print(sdf.shape,pts.shape)
        optimizer.zero_grad()
        loss_sds=sd_model.train_step(text_ebd,color[None].permute(0,3,1,2))
        #loss_bnd=sdf_boundary(sdf,pts)
        #loss_geo=
        #loss_smooth=window_denoise_loss(img_optim)
        loss_all=loss_sds
        loss_all.backward()
        #does it generate details?
        optimizer.step()
        if i%100==0:
            print(i,":",loss_sds)
        #scheduler.step()
        if i%500==0:
            color_=(color)
            img=(color_*255).detach().cpu().numpy().astype(np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./result_nerf/img/{i:03d}.png",img)
        if i%3000==0:
            torch.save(nerf_inside.state_dict(),f"./saved_model_nerf/{i:05d}.pth")
            print(f"saved model at {i} iteration;")