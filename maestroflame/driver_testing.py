'''
To test a pytorch model on a data set.
'''

import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
from glob import glob
import warnings
import sys
import pandas as pd
import optuna
from optuna.trial import TrialState
from datetime import datetime
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEBUG_MODE = True

from reactdataset2 import ReactDataset2

data_paths = ['data/data4/flame/', 'data/data5/flame/']
data_names = ['Dens_fuel = 1e9', 'Temp_fuel = 1e9']
model_path = 'big_1000_run/'
input_prefix = 'react_inputs_*'
output_prefix = 'react_outputs_*'
plotfile_prefix = 'flame_*'


X_data = []
Y_data = []

for data_path in data_paths:

    #load testing loss data to compare
    component_losses_test = np.loadtxt(model_path + 'component_losses_test.txt')
    cost_per_epoch = np.loadtxt(model_path + 'cost_per_epoch.txt')

    last_cost = cost_per_epoch[-1]
    last_comp_loss = component_losses_test[-1, :]

    sfactors = np.loadtxt(model_path + 'scaling_factors.txt')

    #load dataset that we're going to test
    react_data = ReactDataset2(data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=DEBUG_MODE)

    #Normalize with the factors given by the model.
    react_data.input_data[:, 14, :]  = react_data.input_data[:, 14, :]/sfactors[0]
    react_data.input_data[:, 15, :]  = react_data.input_data[:, 15, :]/sfactors[1]
    react_data.output_data[:, 13, :] = react_data.output_data[:, 13, :]/sfactors[2]

    #cut off derivatives, we don't use them in this model make this an option eventually
    Xs = react_data.output_data[:, :13, :]
    enucs = react_data.output_data[:, 26, :]
    react_data.output_data = torch.cat((Xs, enucs.reshape([enucs.shape[0], 1, enucs.shape[1]])), dim=1)

    fields = [field[1] for field in yt.load(react_data.output_files[0])._field_list]

    # print("-----debugging-----")
    # print(type(yt.load(react_data.output_files[0])))
    # print(yt.load(react_data.output_files[0]))
    # print(yt.load(react_data.output_files[0]).field_list)
    fields = fields[:13] + [fields[26]]

    #load model
    from networks import Deep_Net
    model = Deep_Net(react_data.input_data.shape[1], 32, 32, 32, 32, 32, 32, 32, react_data.output_data.shape[1])
    model.load_state_dict(torch.load(model_path + 'my_model.pt'))
    model.eval() #docs said i had to do this


    model.load('mymodel.pt')
    #model.eval()


    #percent cut for testing
    percent_test = 0
    N = len(react_data)

    Num_test  = int(N*percent_test/100)
    Num_train = N-Num_test

    train_set, test_set = torch.utils.data.random_split(react_data, [Num_train, Num_test])

    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)



    # ------------------------ LOSS ---------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    from losses import component_loss_f, log_loss


    criterion = log_loss
    criterion_plotting = nn.MSELoss()


    #plot storage
    cost_per_epoc_new = [] #stores total loss at each epoch
    component_losses_new = [] #stores component wise loss at each epoch (train data)

    losses = []
    plotting_losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):


        # forward
        pred = model(data)
        loss = criterion(pred, targets)

        losses.append(loss.item())


        loss_plot = criterion_plotting(pred, targets)
        plotting_losses.append(loss_plot.item())

        loss_c = component_loss_f(pred, targets)
        loss_c = np.array(loss_c.tolist())
        if batch_idx == 0:
            component_loss = loss_c
        else:
            component_loss = component_loss + loss_c




    component_losses_new = component_loss/batch_idx
    cost_per_epoc_new = sum(plotting_losses) / len(plotting_losses)


    # Relative Loss: |pred-real|/real
    relative_loss = np.abs(cost_per_epoc_new - last_cost)/last_cost

    c_relatives = []
    for i in range(len(component_losses_new)):
        c_relatives.append(np.abs(component_losses_new[i] - last_comp_loss[i])/last_comp_loss[i])


    c_relatives.append(relative_loss)
    fields.append("total cost")


    X_data.append(fields)
    Y_data.append(c_relatives)


import matplotlib.pyplot as plt
import numpy as np



x = np.arange(len(fields))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Y_data[0], width, label=data_names[0])
rects2 = ax.bar(x + width/2, Y_data[1], width, label=data_names[1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Relative Error (assuming the old error is absolute)')
ax.set_xticks(x)
ax.set_xticklabels(fields)
ax.legend()


#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
plt.ylim([0, 10])
fig.tight_layout()

plt.show()


print("Success! :) \n")
