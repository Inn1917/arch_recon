import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import random
import math
import numpy as np
def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 100], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs

PI=3.1415926535
class CamDataset(Dataset):
    def __init__(self,resolution=64,cam_path=None,cam_size=20,cam_type="radial"):
        super().__init__()
        #generate radial cameras;
        self.cam_path=cam_path
        self.cam_size=cam_size
        self.cam_type=cam_type
        self.cam_list=[]
        self.cam_Rt=None
        self.theta=[]
        self.phi=[]
        if self.cam_path is not None:
            cam_list_=os.listdir(cam_path) 
            for i in cam_list_:
                if i[-3:]=="pth":
                    self.cam_list.append(i)
        else:
            for i in range(self.cam_size):
                self.cam_list.append(f"{i:03d}.pth")
            self.prepare_camera()
        #self.K=torch.tensor([[32,0,16],[0,32,16],[0,0,1]],dtype=torch.float32)
        #ss=1.2
        #fine a little bit hard to do so;
        #more regularization;
        f=resolution
        c=f//2
        self.K=torch.tensor([[f,0,c],[0,f,c],[0,0,1]],dtype=torch.float32)
    def prepare_camera(self):
        self.cam_Rt=[]
        for i in range(self.cam_size):
            max_,min_=0.,-0.
            theta=random.random()*2*PI
            phi=(random.random()*(max_-min_)+min_)*PI/2.
            R1=torch.tensor([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[math.cos(theta),0,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]],dtype=torch.float32)
            
            '''R1=torch.tensor([[math.cos(phi),0,-math.sin(phi)],[0,1,0],[math.sin(phi),0,math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[1,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]],dtype=torch.float32)'''
            R=torch.matmul(R1,R2)

            t=torch.tensor([0,0,3.],dtype=torch.float32)
            Rt=torch.zeros((3,4),dtype=torch.float32)
            Rt[:3,:3]=R
            Rt[:3,3]=t
            self.cam_Rt.append(Rt)
            self.theta.append(theta)
            self.phi.append(phi)
        '''pose,_=rand_poses(size=self.cam_size,device="cpu")
        for i in range(self.cam_size):
            self.cam_Rt.append(pose[i,:3])'''
    def __len__(self):
        return len(self.cam_list)
    def __getitem__(self,idx):

        Rt=self.cam_Rt[idx]
        theta,phi=self.theta[idx],self.phi[idx]
        return self.K,Rt,theta,phi
    def save_camera(self,cam_path):
        for i in range(self.cam_size):
            torch.save(self.cam_Rt[i],os.path.join(cam_path,f"{i:03d}.pth"))

class CamDataset_R(Dataset):
    def __init__(self,resolution=64,cam_path=None,cam_size=20,cam_type="radial"):
        super().__init__()
        #generate radial cameras;
        self.cam_path=cam_path
        self.cam_size=cam_size
        self.cam_type=cam_type
        self.cam_list=[]
        self.cam_Rt=None
        self.theta=[]
        self.phi=[]
        if self.cam_path is not None:
            cam_list_=os.listdir(cam_path) 
            for i in cam_list_:
                if i[-3:]=="pth":
                    self.cam_list.append(i)
        else:
            for i in range(self.cam_size):
                self.cam_list.append(f"{i:03d}.pth")
            self.prepare_camera()
        #self.K=torch.tensor([[32,0,16],[0,32,16],[0,0,1]],dtype=torch.float32)
        #ss=1.2
        #fine a little bit hard to do so;
        #more regularization;
        f=resolution
        c=f//2
        self.K=torch.tensor([[f,0,c],[0,f,c],[0,0,1]],dtype=torch.float32)
    def prepare_camera(self):
        self.cam_Rt=[]
        for i in range(self.cam_size):
            max_,min_=0.2,-0.2
            theta=random.random()*2*PI
            phi=(random.random()*(max_-min_)+min_)*PI/2.
            R1=torch.tensor([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[math.cos(theta),0,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]],dtype=torch.float32)
            
            '''R1=torch.tensor([[math.cos(phi),0,-math.sin(phi)],[0,1,0],[math.sin(phi),0,math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[1,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]],dtype=torch.float32)'''
            R=torch.matmul(R1,R2)

            max_1,min_1=0.5,-0.5
            rand_l=(random.random()*(max_1-min_1)+min_1)
            t=torch.tensor([0,0,3.+rand_l],dtype=torch.float32)
            Rt=torch.zeros((3,4),dtype=torch.float32)
            Rt[:3,:3]=R
            Rt[:3,3]=t
            self.theta.append(theta)
            self.phi.append(phi)
            self.cam_Rt.append(Rt)
        '''pose,_=rand_poses(size=self.cam_size,device="cpu")
        for i in range(self.cam_size):
            self.cam_Rt.append(pose[i,:3])'''
    def __len__(self):
        return len(self.cam_list)
    def __getitem__(self,idx):
        if self.cam_path is not None:
            Rt=torch.load(os.path.join(self.cam_path,self.cam_list[idx]))
        else:
            Rt=self.cam_Rt[idx]
        K=self.K.clone()
        theta,phi=self.theta[idx],self.phi[idx]
        return K,Rt,theta,phi
    def save_camera(self,cam_path):
        for i in range(self.cam_size):
            torch.save(self.cam_Rt[i],os.path.join(cam_path,f"{i:03d}.pth"))

class CamDataset_FR(Dataset):
    def __init__(self,resolution=64,cam_path=None,cam_size=20,cam_type="radial"):
        super().__init__()
        #generate radial cameras;
        self.cam_path=cam_path
        self.cam_size=cam_size
        self.cam_type=cam_type
        self.cam_list=[]
        self.cam_Rt=None
        if self.cam_path is not None:
            cam_list_=os.listdir(cam_path) 
            for i in cam_list_:
                if i[-3:]=="pth":
                    self.cam_list.append(i)
        else:
            for i in range(self.cam_size):
                self.cam_list.append(f"{i:03d}.pth")
            self.prepare_camera()
        #self.K=torch.tensor([[32,0,16],[0,32,16],[0,0,1]],dtype=torch.float32)
        #ss=1.2
        #fine a little bit hard to do so;
        #more regularization;
        f=resolution
        c=f//2
        self.K=torch.tensor([[f,0,c],[0,f,c],[0,0,1]],dtype=torch.float32)
    def prepare_camera(self):
        self.cam_Rt=[]
        for i in range(self.cam_size):
            max_,min_=0.2,-0.2
            theta=random.random()*2*PI
            phi=(random.random()*(max_-min_)+min_)*PI/2.
            R1=torch.tensor([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[math.cos(theta),0,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]],dtype=torch.float32)
            
            '''R1=torch.tensor([[math.cos(phi),0,-math.sin(phi)],[0,1,0],[math.sin(phi),0,math.cos(phi)]],dtype=torch.float32)
            R2=torch.tensor([[1,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]],dtype=torch.float32)'''
            R=torch.matmul(R1,R2)
            t=torch.tensor([0,0,3.],dtype=torch.float32)
            Rt=torch.zeros((3,4),dtype=torch.float32)
            Rt[:3,:3]=R
            Rt[:3,3]=t
            self.cam_Rt.append(Rt)
        '''pose,_=rand_poses(size=self.cam_size,device="cpu")
        for i in range(self.cam_size):
            self.cam_Rt.append(pose[i,:3])'''
    def __len__(self):
        return len(self.cam_list)
    def __getitem__(self,idx):
        max_,min_=0.8,-0.
        theta=random.random()*2*PI
        phi=(random.random()*(max_-min_)+min_)*PI/2.
        R1=torch.tensor([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]],dtype=torch.float32)
        R2=torch.tensor([[math.cos(theta),0,-math.sin(theta)],[0,1,0],[math.sin(theta),0,math.cos(theta)]],dtype=torch.float32)
        
        '''R1=torch.tensor([[math.cos(phi),0,-math.sin(phi)],[0,1,0],[math.sin(phi),0,math.cos(phi)]],dtype=torch.float32)
        R2=torch.tensor([[1,0,0],[0,math.cos(theta),math.sin(theta)],[0,-math.sin(theta),math.cos(theta)]],dtype=torch.float32)'''
        R=torch.matmul(R1,R2)
        t=torch.tensor([0,0,3.],dtype=torch.float32)
        Rt=torch.zeros((3,4),dtype=torch.float32)
        Rt[:3,:3]=R
        Rt[:3,3]=t
        ss=random.random()*(0.65)+0.7
        K=self.K.clone()
        K[0,0]*=ss
        K[1,1]*=ss
        return K,Rt,theta,phi
    def save_camera(self,cam_path):
        for i in range(self.cam_size):
            torch.save(self.cam_Rt[i],os.path.join(cam_path,f"{i:03d}.pth"))