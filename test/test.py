#first the geometry then render;
import torch
import numpy as np
from extern.nvdiffrec_core.geometry.dmtet import DMTetGeometry
import argparse
import os
from pyhocon import ConfigFactory
from src.models.triplane_nerf import Triplane_NeRF
import nvdiffrast.torch as dr
from extern.nvdiffrec_core.render.obj import write_obj
from src.utils.ray_sampler import *
from src.dataset.dtu_dataset import MVSDatasetDTU
import cv2
import random
os.environ["CUDA_VISIBLE_DEVICES"]="2"

device="cuda"

if __name__=="__main__":
    Fn=torch.nn.MSELoss()
    dataset=MVSDatasetDTU("/data/yzq/dataset/dtu","train")
    network=Triplane_NeRF().to(device)
    optimizer=torch.optim.Adam(list(network.parameters()),lr=5e-3)
    n=len(dataset)
    img_=torch.zeros(512,640,3,dtype=torch.float32,device=device)
    for i in range(5000):
        for j in range(n):
            chunk=dataset[j]
            K,w2c,nf=chunk["intrinsic"].to(device),chunk["w2c"].to(device),chunk["near_far"].to(device)
            img=chunk["image"].to(device).permute(1,2,0)
            #print(img.shape,img.max(),img.min())
            pts,rays_d,z_vals,gt=ray_sampler(K,w2c,nf,(512,640),2048,96,image=img)
            pts[...,2]-=3.

            out=network(pts,rays_d)
            rgb,depth=raw2outputs(out,z_vals,rays_d)
            loss=Fn(rgb,target=gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i%10==0:
            print(loss,rgb.max(),rgb.min())
        if i%50==0:
            with torch.no_grad():
                ss=random.randint(0,n-1)
                chunk_=dataset[ss]
                K_,w2c_,nf_=chunk_["intrinsic"].to(device),chunk_["w2c"].to(device),chunk_["near_far"].to(device)
                pts_,rays_d_,z_vals_=ray_sampler_full(K_,w2c_,nf_,(512,640),96)
                pts_[...,2]-=3.
                sz=2560
                for k in range(512*640//sz):
                    out_=network(pts_[sz*k:sz*(k+1)],rays_d_[sz*k:sz*(k+1)])
                    rgb_,_=raw2outputs(out_,z_vals_[sz*k:sz*(k+1)],rays_d_[sz*k:sz*(k+1)])
                    rgb_=rgb_.reshape(sz//640,640,3)
                    img_[(sz//640)*k:(sz//640)*(k+1)]=rgb_
                img_[img_<0.]=0.
                img_[img_>1.]=1.
                img_np=(img_.cpu().numpy()*255.).astype(np.uint8)
                img_nb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"/data/yzq/Lab_Task/arch_recon/result/img/imperial_{i:03d}.png",img_nb)
                