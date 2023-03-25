from src.modules.common_import import *

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            #x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

def norm_mlp(x):
    m=x.mean(dim=-1,keepdim=True)
    v=((x-m)*(x-m)).mean(dim=-1,keepdim=True)
    return (x-m)/(v+1e-4)
class SimpleMLP(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_dim = 128
        self.ln=nn.LayerNorm([self.hidden_dim])
        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=1.),
            self.ln,#torch.nn.Softplus(),
            FullyConnectedLayer( self.hidden_dim, self.hidden_dim, lr_multiplier=1.),
            self.ln,#torch.nn.Softplus(),
            FullyConnectedLayer( self.hidden_dim, self.hidden_dim, lr_multiplier=1.),
            self.ln,#torch.nn.Softplus(),
            FullyConnectedLayer( self.hidden_dim, self.hidden_dim, lr_multiplier=1.),
            self.ln,#torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + 3, lr_multiplier=1.)
        )
        self.activation = torch.nn.ReLU()
    def forward(self, sampled_features,d):
        # Aggregate features
        #sampled_features = sampled_features.mean(1)
        x = torch.cat([sampled_features,d],dim=-1)

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        
        '''rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        print(rgb.max(),rgb.min())
        sigma = self.activation(x[..., 0:1])'''
        return x