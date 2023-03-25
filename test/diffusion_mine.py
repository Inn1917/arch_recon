import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from denoising_diffusion_pytorch import Unet
from math import sqrt
import os
#so it is some strange problem of the freaking keyboard...
#then what? some kind of generation?
def linear_beta_schedule(tstep):
    scale=1000/tstep
    min_=scale*1e-4
    max_=scale*2e-2
    beta=torch.linspace(min_,max_,steps=tstep)
    return beta

#alright, so...what now?
#detail enhancement, by camera recalibrate;
#diffusion model loss, effective;
#look into the diffusion model with condition;
#nice;
class DiffusionModel(nn.Module):
    def __init__(self,input_dim=32,timestep=50):
        # a diffusion model with unet as per-level predictor;
        super().__init__()
        self.register_buffer("beta_t",linear_beta_schedule(timestep))
        self.register_buffer("variance_t",linear_beta_schedule(timestep))
        self.network=Unet(dim = input_dim,dim_mults = (1, 2, 4, 8))
        self.tstep=timestep
        self.input_dim=input_dim
        '''self.alpha_t=[(1-self.beta_t)]
        for i in range(self.tstep-1):
            self.alpha_t.append(self.alpha_t[-1]*(1-self.beta_t))'''
        self.register_buffer("alpha_t",torch.cumprod(1.-self.beta_t,dim=0))
        
        #fine then;
    def forward(self,x0,t):
        #use this function to train the network;
        #x0:[b,3,n,n];float
        #t:[b];long
        bsize=x0.shape[0]
        epsilon=torch.randn((bsize,3,self.input_dim,self.input_dim)).to(self.beta_t.device)
        mean=torch.sqrt(self.alpha_t[t])[:,None,None,None]*x0+torch.sqrt(1.-self.alpha_t[t])[:,None,None,None]*epsilon
        pred_val=self.network(mean,t)
        return pred_val,epsilon
    def pred(self):
        with torch.no_grad():
            xt=torch.randn((1,3,self.input_dim,self.input_dim)).to(self.beta_t.device)
            for i in range(self.tstep-1,-1,-1):
                t=torch.tensor([i]).long().to(self.beta_t.device)
                z=torch.randn((1,3,self.input_dim,self.input_dim)).to(self.beta_t.device) if i>0 else 0.
                #print(self.alpha_t.min(),self.alpha_t.max())
                xt=1./torch.sqrt(1.-self.beta_t[i])[None][:,None,None,None]*(xt-(self.beta_t[i])[None][:,None,None,None]/torch.sqrt(1.-self.alpha_t[i])[None][:,None,None,None]*self.network(xt,t))+torch.sqrt(self.variance_t[i])[None][:,None,None,None]*z
            return xt

def train_image(model_path):
    device="cuda:2"
    model=DiffusionModel(input_dim=128,timestep=50).to(device)
    model.load_state_dict(torch.load(model_path))
    img1=cv2.imread("/disk2/yzq/Lab_Task/diffusion/folder/beast_sampai.webp")
    img1=cv2.resize(img1,(128,128))
    cv2.imwrite("/disk2/yzq/Lab_Task/diffusion/test1.png",img1)
    img_t=(2.*(torch.from_numpy(img1).float()/255.-0.5))[None].permute(0,3,1,2).to(device)

    randimg=(torch.rand((1,3,128,128))*2.-1.).to(device)
    lr=1e-3
    with torch.no_grad():
        for i in range(114514):
            t=torch.randint(50,size=(1,)).to(device)
            pred,eps=model.forward(randimg,t)
            grad=pred-eps
            randimg-=grad*lr
            if i%100==0:
                res=randimg.detach().cpu()
                max_,min_=res.max(),res.min()
                print(max_,min_)
                #res=2.*func((res+1.)/2.)-1.
                res[res>1.]=1.
                res[res<-1.]=-1.

                res_=((res+1.)/2.*255.)[0].permute(1,2,0).numpy().astype(np.uint8)
                print(res_.shape,np.max(res_),np.min(res_))
                cv2.imwrite("/disk2/yzq/Lab_Task/diffusion/test1.png",res_)

if __name__=="__main__":
    #so the time is long;
    #then what?
    import os
    train_image("/disk2/yzq/Lab_Task/diffusion/savedir/5300.pth")