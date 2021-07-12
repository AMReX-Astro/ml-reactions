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

yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
import gc

yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
import gc

#from reactdataset import ReactDataset
from reactdataset2 import ReactDataset2
from plotting import make_movie

data_path = '../data1/flame/'
input_prefix = 'react_inputs_*'
output_prefix = 'react_outputs_*'
plotfile_prefix = 'flame_*'


plotfiles = glob(data_path + plotfile_prefix)
plotfiles = sorted(plotfiles)
plotfiles = plotfiles[:-2] #cut after divuiter and initproj
plotfiles = [plotfiles[-1]] + plotfiles[:-1] #move initdata to front.
#make_movie(plotfiles, movie_name='enuc.mp4', var='enuc')


react_data = ReactDataset2(data_path, input_prefix, output_prefix, plotfile_prefix)

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

class Net(nn.Module):
    def __init__(self, input_size, h1, h2, h3, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.ReLU()
        self.fc4 = nn.Linear(h3, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.fc4(x)
        return x


net = Net(16, 16, 32, 16, 15)

# Hyperparameters
num_classes = 26
learning_rate = 1e1
batch_size = 5
num_epochs = 30
input_size = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model
model = Net(react_data.input_data.shape[1], 64, 128, 64, react_data.output_data.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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



# ------------------------ HYPERPARAMETER OPTIMIZATION ----------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


DEVICE = torch.device("cpu")
BATCHSIZE = 16
CLASSES = 15
DIR = os.getcwd()
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

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", LAYERS[0], LAYERS[1])

    layers = []

    in_features = 16
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), UNITS[0], UNITS[1])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), DROPOUT_RATE[0], DROPOUT_RATE[1])
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZERS)
    lr = trial.suggest_float("lr", LEARNING_RATE[0], LEARNING_RATE[1], log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)



    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break


            optimizer.zero_grad()
            output = model(data)
            L = nn.MSELoss()
            loss = L(output, target)
            loss.backward()
            optimizer.step()


        #Validation of model
        model.eval()
        #we save the largest MSE of the set as the accuracy of the trial.
        accuracy = -np.inf
        L = nn.MSELoss()
        with torch.no_grad():
             for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)


                    local_accuracy = L(output, target)

                    if local_accuracy > accuracy:
                        accuracy = local_accuracy

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
