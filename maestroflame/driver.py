import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import Video
from glob import glob
import torch
import warnings
import sys
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.trial import TrialState
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import gc
import matplotlib.cbook

DEBUG_MODE = True


yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations

yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
import gc

#from reactdataset import ReactDataset
from reactdataset2 import ReactDataset2
from plotting import make_movie

data_path = 'data/data1/flame/'
input_prefix = 'react_inputs_*'
output_prefix = 'react_outputs_*'
plotfile_prefix = 'flame_*'


plotfiles = glob(data_path + plotfile_prefix)
plotfiles = sorted(plotfiles)
plotfiles = plotfiles[:-2] #cut after divuiter and initproj
plotfiles = [plotfiles[-1]] + plotfiles[:-1] #move initdata to front.
#make_movie(plotfiles, movie_name='enuc.mp4', var='enuc')


react_data = ReactDataset2(data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=DEBUG_MODE)

#Normalize density, temperature, and enuc
dens_fac = torch.max(react_data.input_data[:, 14, :])
temp_fac = torch.max(react_data.input_data[:, 15, :])
enuc_fac = torch.max(react_data.output_data[:, 14, :])

react_data.input_data[:, 14, :]  = react_data.input_data[:, 14, :]/dens_fac
react_data.input_data[:, 15, :]  = react_data.input_data[:, 15, :]/temp_fac
react_data.output_data[:, 14, :] = react_data.output_data[:, 14, :]/enuc_fac

react_data.cut_data_set(2)



# ------------------------ NEURAL NETWORK -----------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

#percent cut for testing
percent_test = 10
N = len(react_data)

Num_test  = int(N*percent_test/100)
Num_train = N-Num_test

train_set, test_set = torch.utils.data.random_split(react_data, [Num_train, Num_test])

train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)

from networks import Net, OC_Net

net = Net(16, 16, 32, 16, 15)

# Hyperparameters
num_classes = 26
learning_rate = 1e1
batch_size = 5
if DEBUG_MODE:
    num_epochs = 5
else:
    num_epochs = 30
input_size = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




BATCHSIZE = 16
CLASSES = 15
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
#hyperparameter space
LAYERS = [1,5]
UNITS = [4, 128] #nodes per layer
DROPOUT_RATE = [0, 0.5]
LEARNING_RATE = [1e-7, 1e-1]
OPTIMIZERS = ["Adam", "RMSprop", "SGD"]
#optimizer study

n_trials=5
timeout=600

from hyperparamter_optimization import do_h_opt
hyper_results = do_h_opt(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
                      LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                      LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                      n_trials, timeout)


#Model
model = Net(react_data.input_data.shape[1], 64, 128, 64, react_data.output_data.shape[1])

model = OC_Net(react_data.input_data.shape[1], react_data.output_data.shape[1], hyper_results)

optimizer = hyper_results['optimizer']
lr = hyper_results['lr']
if optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    print("unsupported optimizer, please define it.")
    sys.exit()



# Loss and optimizer
criterion = nn.MSELoss()

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
