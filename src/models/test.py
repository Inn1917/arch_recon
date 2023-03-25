import torch
import torch.nn as nn
import torch.nn.functional as F
import os
print(os.getcwd())
from hashencoding_sdf import HashEncoding_SDF


os.environ["CUDA_VISIBLE_DEVICES"]="2"
if __name__=="__main__":
    dev="cuda"
    model=HashEncoding_SDF().to(dev)
    x=torch.rand((10,100,3)).to(dev)
    y=model(x)
    print(y.shape)
