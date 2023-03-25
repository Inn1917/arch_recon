from mlp_nv import SimpleMLP
from hashencoding_nv import SimpleHashEncoding
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__=="__main__":
    '''
    
    dev="cuda"
    model=SimpleMLP(3,6,128,5).to(dev)
    optim=torch.optim.Adam(model.parameters(),lr=1e-2)
    x=torch.rand((10,5,3)).to(dev)
    gt=torch.rand((10,5,6)).to(dev)
    for i in range(1000):
        y=model(x)
        loss=torch.abs(y-gt).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i%10==0:
            print(loss)
        
    '''
    
    dev="cuda"
    model=SimpleHashEncoding().to(dev)

    x=torch.rand((10,5,3)).to(dev)

    #for i in range(1000):
    y=model(x)
    print(y.shape)    