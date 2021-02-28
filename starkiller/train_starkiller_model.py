#!/usr/bin/env python
import torch
from ReactionsSystem import ReactionsSystem
from NetTraining import NetTraining

# Set use_cuda=True to use an available GPU
use_cuda=False

# size of training set
NumSamples = 256

system = ReactionsSystem()

x = torch.unsqueeze(torch.linspace(0, system.end_time, NumSamples, requires_grad=True), dim=1)
x_test = torch.unsqueeze(torch.rand(NumSamples, requires_grad=False), dim=1) * system.end_time

# activations, e.g. F.celu, torch.tanh
activations = {}
hidden_depth = 10

for h in range(hidden_depth+1):
    activations[h] = torch.tanh
#    if h < hidden_depth/2:
#        activation[h] = torch.tanh
#    else:
#        activation[h] = F.celu


training = NetTraining(system, x, x_test, use_cuda=use_cuda)
training.init_net(n_i=1, n_d=system.numDependent,
                  n_h=(system.numDependent)*2, depth_h=hidden_depth,
                  activations=activations)

training.train_error(10000)
training.save_history()