import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts.to("cuda:0")).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    #print(weights.device,"dsa")
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples,device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples],device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeRFRenderer:
    def __init__(self,
                 nerf,
                 nerf_inside,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.nerf_inside = nerf_inside
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        with torch.cuda.device(0):
            batch_size, n_samples = z_vals.shape

            # Section length
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.cuda.FloatTensor([sample_dist]).expand(dists[..., :1].shape)], -1)
            mid_z_vals = z_vals + dists * 0.5

            # Section midpoints
            #print(rays_o.shape,rays_d.shape,mid_z_vals.shape)
            pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3


            dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

            pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
            dirs = dirs.reshape(-1, 3)

            density, sampled_color = nerf(pts, dirs)
            alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
            alpha = alpha.reshape(batch_size, n_samples)
            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).to("cuda"), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            sampled_color=torch.sigmoid(sampled_color)
            sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
            color = (weights[:, :, None] * sampled_color).sum(dim=1)
            if background_rgb is not None:
                color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

            return {
                'color': color,
                'sampled_color': sampled_color,
                'alpha': alpha,
                'weights': weights,
            }


    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        with torch.cuda.device(0):
            batch_size = len(rays_o)
            sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
            z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
            z_vals = near + (far - near) * z_vals[None, :]

            z_vals_outside = None
            if self.n_outside > 0:
                z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside).to(rays_o.device)

            n_samples = self.n_samples
            perturb = self.perturb

            if perturb_overwrite >= 0:
                perturb = perturb_overwrite
            if perturb > 0:
                t_rand = (torch.rand([batch_size, 1]) - 0.5).to(rays_o.device)
                z_vals = z_vals + t_rand * 2.0 / self.n_samples

                if self.n_outside > 0:
                    mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1]).to(rays_o.device)
                    upper = torch.cat([mids, z_vals_outside[..., -1:]], -1).to(rays_o.device)
                    lower = torch.cat([z_vals_outside[..., :1], mids], -1).to(rays_o.device)
                    t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]]).to(rays_o.device)
                    #print(lower.device,upper.device,t_rand.device,rays_o.device)
                    z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

            if self.n_outside > 0:
                z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

            background_alpha = None
            background_sampled_color = None

            # currently no upsample
            

            # Background model
            if self.n_outside > 0:
                z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
                z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
                ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

                background_sampled_color = ret_outside['sampled_color']
                background_alpha = ret_outside['alpha']

            # Render core
            ret_fine = self.render_core_outside(rays_o, rays_d, z_vals, sample_dist, self.nerf_inside)

            color_fine = ret_fine['color']
            weights = ret_fine['weights']
            weights_sum = weights.sum(dim=-1, keepdim=True)

            return {
                'color_fine': color_fine,
                'weight_sum': weights_sum,
                'weights': weights
            }
    def forward(self,K,Rt,resolution=(64,64)):
        device=K.device
        h,w=resolution
        img_grid=torch.ones((h,w,3),dtype=torch.float32,device=device)
        img_grid[:,:,0]=torch.linspace(0,w-1,steps=w)[None,:]
        img_grid[:,:,1]=torch.linspace(0,h-1,steps=h)[:,None]

        rays_d=img_grid.reshape(-1,3)
        
        N=rays_d.shape[0]
        rays_o=torch.zeros((N,3),dtype=torch.float32,device=device)
        rays_o[...,:]=-torch.matmul(Rt[:3,:3].transpose(1,0),Rt[:3,3:])[:,0][None,:]
        
        K_inv=K.clone().detach()
        K_inv[0,0]=1./K_inv[0,0]
        K_inv[1,1]=1./K_inv[1,1]
        K_inv[0,2]=K_inv[0,0]*K_inv[0,2]
        K_inv[1,2]=K_inv[1,1]*K_inv[1,2]
        K_inv[:2,2]*=-1.
        #print(K_inv)
        rays_d=torch.matmul(K_inv[None],rays_d[...,None])
        rays_d=torch.matmul(Rt[:3,:3].transpose(1,0),rays_d)[...,0]
        near=1.
        far=3.
        result=self.render(rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0)
        color=result["color_fine"]
        return color.reshape(h,w,3)
    #def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
    #    return extract_geometry(bound_min,
    #                            bound_max,
    #                            resolution=resolution,
    #                            threshold=threshold,
    #                            query_func=lambda pts: -self.nerf_inside.(pts))
