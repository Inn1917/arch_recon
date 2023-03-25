import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extern.diffusion_model.sd import StableDiffusion
from src.renderer.trimesh_renderer import TriMesh_Renderer
from src.dataset.rand_cam import CamDataset
import random
#def get_geo_loss(vtx,tri_idx):
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def window_denoise_loss(img):
    p1=img[1:-1,1:-1]
    p2=img[1:-1,:-2]
    p3=img[1:-1,2:]
    p4=img[:-2,1:-1]
    p5=img[2:,1:-1]
    p6=img[:-2,:-2]
    p7=img[:-2,2:]
    p8=img[2:,:-2]
    p9=img[2:,2:]
    loss_=(p1+p2+p3+p4+p5+p6+p7+p8+p9)/9.-p1
    return (loss_*loss_).sum()
if __name__=="__main__":
    import os
    import cv2
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
    opt = parser.parse_args()

    renderer=TriMesh_Renderer(opt).to(device)

    
    sd_model=StableDiffusion(device)
    text_ebd=sd_model.get_text_embeds(["a high quality photo of a pineapple"],[""])

    img=cv2.imread("./dummy_dataset/sample_0.jpg")
    img=cv2.resize(img,(512,512))
    img_blur = cv2.blur(img,(7,7))
    img=torch.from_numpy(img.astype(np.float32)).to(device)
    img=img/255.



    img_optim=torch.randn((512,512,3),dtype=torch.float32,device=device)*1e-1
    img_optim=torch.from_numpy(img_blur.astype(np.float32)/255.).to(device)
    img_optim.requires_grad_(True)
    #optimizer=torch.optim.Adam(renderer.mesh.parameters(),lr=5e-3)
    optimizer=torch.optim.Adam([img_optim],lr=5e-3)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,80], gamma=0.5)


    img_cmp=img[None].permute(0,3,1,2)
    for i in range(opt.max_iter):
        K,Rt=dataset[i%L]
        K=K.to(device)
        Rt=Rt.to(device)
        color=torch.sigmoid(img_optim[None])#renderer.forward(K,Rt)

        optimizer.zero_grad()
        loss_sds=sd_model.train_step(text_ebd,color.permute(0,3,1,2))
        #loss_geo=
        #loss_smooth=window_denoise_loss(img_optim)
        loss_all=loss_sds#+10.*loss_smooth
        loss_all.backward()

        optimizer.step()
        if i%100==0:
            print(i,":",loss_sds)
            scheduler.step()
        if i%500==0:
            color_=(color)
            img=(color_[0]*255).detach().cpu().numpy().astype(np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"./result/{i:03d}.png",img)