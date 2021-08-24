#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

import sys
sys.path.insert(-1,'../../maestroflame')

from networks import Net, OC_Net

## Read Model file

filename = "example_model.pt"

model = Net(16, 64, 128, 64, 14)

model.load_state_dict(torch.load(filename))
# model.eval()

print(model)

## Converting to Torch Script (Annotation)

# Using annotation
net_module = torch.jit.script(model)
net_module.save("ts_model.pt")
print(net_module.code)
