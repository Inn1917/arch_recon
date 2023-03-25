from src.modules.common_import import *

class SimpleHashEncoding(nn.Module):
    def __init__(self,num_levels=16,hashmap_size=19,base_resolution=16,hash=True,smoothstep=True):
        super().__init__()
        self.config_encoding={
			"otype": "HashGrid" if hash else "DenseGrid",
			"n_levels": num_levels,
			"n_features_per_level": 2,
			"log2_hashmap_size": hashmap_size,
			"base_resolution": base_resolution,
			"per_level_scale": 1.5,
			"interpolation": "Smoothstep" if smoothstep else "Linear"
		}
        self.network=tcnn.Encoding(3,self.config_encoding)
    def forward(self,x):
        #x:[b,d,3]
        if len(x.shape)==2:
            return self.network(x)
        else:
            b,d,c=x.shape
            x=x.reshape(b*d,c)
            y=self.network(x)
            y=y.reshape(b,d,-1)
            return y
