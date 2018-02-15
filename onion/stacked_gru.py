import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F


def residual(f, x, *args):
    fx, _ = f(x, *args)
    return x + fx

class StackedGRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, residual=False, **kwargs):
        super(StackedGRU, self).__init__()
        assert num_layers > 0
        util.autoassign(locals())
        self.bottom = nn.GRU(input_size, hidden_size, 1, **kwargs)
        self.layers = nn.ModuleList()

     
        for i in range(num_layers-1):
            layer = nn.GRU(hidden_size, hidden_size, 1, **kwargs)
            self.layers.append(layer)

    def forward(self, x, h0):
        out, _ = self.bottom(x, h0[0:1,:,:])
        for i in range(self.num_layers-1):
            if self.residual: 
                out =  residual(self.layers[i], out, h0[i+1:i+2,:,:])
            else:
                out, _ = self.layers[i](out, h0[i+1:i+2,:,:])
        return out
        
    