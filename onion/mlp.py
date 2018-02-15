import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, sizes, activation=F.tanh):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()


        for size_in, size_out in zip(sizes[:-1], sizes[1:]):
            self.layers.append(nn.Linear(size_in, size_out))
    
    def forward(self, X):
        for i in range(0, len(self.layers)-1):
            X = self.activation(self.layers[i](X))
        return self.layers[-1](X)

