
from torch.utils.data import Dataset
import sys 
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T


class MVSDatasetDTU(Dataset):
    def __init__(self, root_dir, split, levels=1, img_wh=None, downSample=1.0, max_len=-1,use_metas=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train", "val" or "test"!'
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.build_metas()
        self.levels = levels  # FPN levels
        self.build_proj_mats()
        self.define_transforms()
        self.use_metas=use_metas
        
        print(f'==> image down scale: {self.downSample}')

    '''def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])
    '''
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    ])

    def build_metas(self):
        self.metas = []
        
        str_=f'./src/dataset/dset_extern/dtu_{self.split}.txt'

        
        with open(str_) as f:
            self.scans = [line.rstrip() for line in f.readlines()]


        self.scene_num=len(self.scans)
        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3]
        # a total of 31 views;
        self.id_list = []

        for scan in self.scans:
            with open(f'./src/dataset/dset_extern/dtu_pairs.txt') as f:
                
                num_viewpoint = int(f.readline())
                
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
                        self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()
        
    def build_proj_mats(self):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        near_fars=[]
        for vid in self.id_list:

            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor
            #the full scale image of size 512,640;
            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics += [intrinsic.copy()]

            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]
            near_fars.append(np.array(near_far))
        
        self.near_fars=np.stack(near_fars)
        self.intrinsics = np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]

    #they got their things to do;
    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i
    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len
    
    def __getitem__(self, idx):
        sample = {}
        #print("index:\n",idx)#[247 error;]
        scan, light_idx, target_view, src_views = self.metas[idx]

        
        # NOTE that the id in image file names is from 1 to 49 (not 0~48)
        img_filename = os.path.join(self.root_dir,
                                    f'Rectified/{scan}_train/rect_{target_view + 1:03d}_{light_idx}_r5000.png')
        
        #now we move on;
        img = Image.open(img_filename)
        img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
        img = img.resize(img_wh, Image.BILINEAR)
        img = self.transform(img)
        

        index_mat = self.remap[target_view]
        
        near_far = self.near_fars[index_mat]
        intrinsic=(self.intrinsics[index_mat])
        w2c=(self.world2cams[index_mat])
        c2w=(self.cam2worlds[index_mat])


        '''imgs_a=torch.stack(imgs_a).float()
        intrinsics_a, w2cs_a, c2ws_a = np.stack(intrinsics_a), np.stack(w2cs_a), np.stack(c2ws_a)'''

        sample['image'] = img  # (H, W, 3)
        sample['w2c'] = torch.from_numpy(w2c.astype(np.float32))  # (4, 4)
        sample['c2w'] = torch.from_numpy(c2w.astype(np.float32))  # (4, 4)
        sample['near_far'] = torch.from_numpy(near_far.astype(np.float32))#[2]
        sample['intrinsic'] = torch.from_numpy(intrinsic.astype(np.float32))  # (V, 3, 3)
        
        return sample

if __name__=="__main__":
    from kornia.utils import create_meshgrid
    from utils import write_pointcloud,compose_training_idx_val,compose_training_idx
    from bounding_box.bounding_box import Rect_BBox_Sim
    dataset=MVSDatasetDTU("/data2/yzq/dtu","val",use_gru=True,use_metas=True)
    bbox=dataset.bbox
    print("bbox check:",bbox)
    bbox=Rect_BBox_Sim(bbox,256,8)#ok;
    mask3d=torch.load("/data2/yzq/dtu/Misc_2/scan42/3dmask.pth").cpu()
    grid=bbox.get_wrd_grid("cpu")
    pts=grid[mask3d]*200
    write_pointcloud("./mask.ply",pts)

