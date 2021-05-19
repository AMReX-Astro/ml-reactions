import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from TrainingHistory import TrainingHistory

# DenseNet has dense layers whose sizes change as follows:
# (n_hidden, n_hidden-dn, n_hidden-2*dn, ..., n_hidden/2, n_hidden/2+dn, ..., n_hidden)
class DenseNet(nn.Module):
    def __init__(self, n_independent, n_dependent,
                 n_hidden, hidden_depth, activation):
        super(DenseNet, self).__init__()
        
        self.activation = activation
        self.input_layer = nn.Linear(n_independent, n_hidden)
        
        self.hidden_layers = nn.ModuleList()
        dn = n_hidden//hidden_depth
        half_depth = hidden_depth//2
        for i in range(half_depth):
            self.hidden_layers.append(nn.Linear(n_hidden-i*dn, n_hidden-(i+1)*dn))
        
        if hidden_depth%2 == 1:
            self.hidden_layers.append(nn.Linear(n_hidden-half_depth*dn, n_hidden-half_depth*dn))
            
        for i in range(half_depth, 0, -1):
            self.hidden_layers.append(nn.Linear(n_hidden-i*dn, n_hidden-(i-1)*dn))
        
        self.output_layer = nn.Linear(n_hidden, n_dependent)
        
        # dropout layer (if needed)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation[0](x)
        
        for i, h in enumerate(self.hidden_layers):
            x = self.activation[i+1](h(x))
        
        # x = self.dropout(x)
        x = self.output_layer(x)
        return x

