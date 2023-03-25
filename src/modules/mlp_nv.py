from src.modules.common_import import *
'''

modules include some basic network structures,the blocks.
which takes only one single input and returns one single output;
the input are processed with positional encoding or normalization beforhead;

'''
class SimpleMLP(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,num_layers):
        super().__init__()
        self.config_network={
            "otype": "CutlassMLP",    
            "activation": "ReLU",        
            "output_activation": "None", 
            "n_neurons": (hidden_dim),           
                                        
            "n_hidden_layers": (num_layers),
        }
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.network=tcnn.Network(input_dim,output_dim,self.config_network)
    def forward(self,x):
        #x:[b,d,c]
        if len(x.shape)==2:
            return self.network(x)
        else:
            b,d,c=x.shape
            x=x.reshape(b*d,c)
            y=self.network(x)
            y=y.reshape(b,d,self.output_dim)
            return y
