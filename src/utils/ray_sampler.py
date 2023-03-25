import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.modules.mlp_nv import SimpleMLP as MLP
from src.modules.neus_embedder import get_embedder

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    device=raw.device
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)#like some kind of inline function;

    dists = z_vals[...,1:] - z_vals[...,:-1]
    #should be n_sample-1;
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)  # [N_rays, N_samples]
    #last one as 1e10; background;
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    #well,it is it;the true distance by all;
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    #should be density itself;
    #so the density range should be between 0 and 1;
    #then what?should be that,smaller than one;
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, depth_map
def ray_sampler(k,w2c,near_far,resolution,ray_num,sample_num,image=None,device="cuda"):
    h,w=resolution
    pixel_grid=torch.ones(h,w,3,dtype=torch.float32,device=device)
    pixel_grid[:,:,0]=torch.linspace(0,w-1,w,device=device)[None]
    pixel_grid[:,:,1]=torch.linspace(0,h-1,h,device=device)[:,None]

    k_inv=k.inverse()
    cam_dir=torch.matmul(k_inv[None,None,:,:],pixel_grid[...,None])
    wrd_dir=torch.matmul(w2c[:3,:3].transpose(1,0)[None,None],cam_dir)[...,0]#[h,w,3]
    cam_center=-torch.matmul(w2c[:3,:3].transpose(1,0),w2c[:3,3:])[:,0]#[3]

    near,far=near_far
    sample_range=torch.linspace(0,1.,sample_num,device=device)*(far-near)+near#[d]

    idx=torch.randperm(h*w,device=device)[:ray_num]
    pts=(wrd_dir[:,:,None,:]*sample_range[None,None,:,None]+cam_center).reshape(-1,sample_num,3)[idx]
    rays_d=wrd_dir.reshape(-1,3)[idx]
    z_vals=sample_range[None,:].repeat(pts.shape[0],1)
    if image is not None:
        gt=image.reshape(-1,3)[idx]
    else:
        gt=None
    return pts,rays_d,z_vals,gt
def ray_sampler_full(k,w2c,near_far,resolution,sample_num,image=None,device="cuda"):
    h,w=resolution
    pixel_grid=torch.ones(h,w,3,dtype=torch.float32,device=device)
    pixel_grid[:,:,0]=torch.linspace(0,w-1,w,device=device)[None]
    pixel_grid[:,:,1]=torch.linspace(0,h-1,h,device=device)[:,None]

    k_inv=k.inverse()
    cam_dir=torch.matmul(k_inv[None,None,:,:],pixel_grid[...,None])
    wrd_dir=torch.matmul(w2c[:3,:3].transpose(1,0)[None,None],cam_dir)[...,0]#[h,w,3]
    cam_center=-torch.matmul(w2c[:3,:3].transpose(1,0),w2c[:3,3:])[:,0]#[3]

    near,far=near_far
    sample_range=torch.linspace(0,1.,sample_num,device=device)*(far-near)+near#[d]

    
    pts=(wrd_dir[:,:,None,:]*sample_range[None,None,:,None]+cam_center).reshape(-1,sample_num,3)
    rays_d=wrd_dir.reshape(-1,3)
    z_vals=sample_range[None,:].repeat(pts.shape[0],1)
    return pts,rays_d,z_vals


