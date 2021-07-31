import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
from IPython.display import Video
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
DO_PLOTTING = True
SAVE_MODEL = True
log_loss_option = True
DO_HYPER_OPTIMIZATION = True


# ------------------------ DATA --------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


#from reactdataset import ReactDataset
from reactdataset2 import ReactDataset2
from plotting import make_movie

data_path = 'data/data1/flame/'
input_prefix = 'react_inputs_*'
output_prefix = 'react_outputs_*'
plotfile_prefix = 'flame_*'
output_dir = 'test_logger/'
log_file = output_dir + "log.txt"

isdir = os.path.isdir(output_dir)
if not isdir:
    os.mkdir(output_dir)



class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()

now = datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print(f"Model starting on : {dt_string}")
print(f"input_prefix {input_prefix}")
print(f"output_prefix {output_prefix}")
print(f"output_dir {output_dir}")
print(f"DEBUG_MODE {DEBUG_MODE}")
print(f"DO_PLOTTING {DO_PLOTTING}")
print(f"log_loss_option {log_loss_option}")
print(f"DO_HYPER_OPTIMIZATION {DO_HYPER_OPTIMIZATION}")



plotfiles = glob(data_path + plotfile_prefix)
plotfiles = sorted(plotfiles)
plotfiles = plotfiles[:-2] #cut after divuiter and initproj
plotfiles = [plotfiles[-1]] + plotfiles[:-1] #move initdata to front.
#make_movie(plotfiles, movie_name='enuc.mp4', var='enuc')


react_data = ReactDataset2(data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=DEBUG_MODE)

#Normalize density, temperature, and enuc
dens_fac = torch.max(react_data.input_data[:, 14, :])
temp_fac = torch.max(react_data.input_data[:, 15, :])
enuc_fac = torch.max(react_data.output_data[:, 13, :])

react_data.input_data[:, 14, :]  = react_data.input_data[:, 14, :]/dens_fac
react_data.input_data[:, 15, :]  = react_data.input_data[:, 15, :]/temp_fac
react_data.output_data[:, 13, :] = react_data.output_data[:, 13, :]/enuc_fac


#percent cut for testing
percent_test = 10
N = len(react_data)

Num_test  = int(N*percent_test/100)
Num_train = N-Num_test

train_set, test_set = torch.utils.data.random_split(react_data, [Num_train, Num_test])

train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)


# ------------------------ NEURAL NETWORK -----------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
from networks import Net, OC_Net


if DEBUG_MODE:
    num_epochs = 5
else:
    num_epochs = 80


if DO_HYPER_OPTIMIZATION:
    BATCHSIZE = 16
    CLASSES = react_data.output_data.shape[1]
    in_features = react_data.input_data.shape[1]
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
    if DEBUG_MODE:
        n_trials = 5
    else:
        n_trials=100
    timeout=600

    from hyperparamter_optimization import do_h_opt, print_h_opt_results
    hyper_results = do_h_opt(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
                          LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                          LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                          n_trials, timeout, in_features)

    print_h_opt_results(hyper_results)

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

else:

    model = Net(react_data.input_data.shape[1], 64, 128, 64, react_data.output_data.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-5)


# ------------------------ LOSS ---------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
from losses import component_loss_f

# Loss and optimizer
if log_loss_option:
    from losses import log_loss
    criterion = log_loss
else:
    criterion = nn.MSELoss()

#As we test different loss functions, its important to keep a consistent one when
#plotting or else we have no way of comparing them.
criterion_plotting = nn.MSELoss()



#plot storage
cost_per_epoc = [] #stores total loss at each epoch
component_losses_test = [] #stores component wise loss at each epoch (test data)
component_losses_train = [] #stores component wise loss at each epoch (train data)
cost_per_epoc_test = []
# Train Network

#try: #try block so you can control c out of it and the model will still be saved.
for epoch in range(num_epochs):
    losses = []
    plotting_losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        with torch.no_grad():
            loss_plot = criterion_plotting(scores, targets)
            plotting_losses.append(loss_plot.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()



    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
    #Cost per epoc

    if DO_PLOTTING:
        with torch.no_grad():
            cost_per_epoc.append(sum(plotting_losses) / len(plotting_losses))


            #Evaulate NN on testing data.
            for batch_idx, (data, targets) in enumerate(test_loader):
                # forward
                scores = model(data)
                loss = criterion_plotting(scores, targets)
                losses.append(loss.item())

            cost_per_epoc_test.append(sum(losses) / len(losses))



            #Component wise error testing data
            for batch_idx, (data, targets) in enumerate(test_loader):
                pred = model(data)
                loss = component_loss_f(pred, targets)
                if batch_idx == 0:
                    component_loss = loss
                else:
                    component_loss = component_loss + loss
            component_losses_test.append(component_loss/batch_idx)

            #Component wise error training data
            for batch_idx, (data, targets) in enumerate(train_loader):
                pred = model(data)
                loss = component_loss_f(pred, targets)
                if batch_idx == 0:
                    component_loss = loss
                else:
                    component_loss = component_loss + loss
            component_losses_train.append(component_loss/batch_idx)

# except KeyboardInterrupt:
#     #plotting will error most likely if this doesn't finish properly
#     DO_PLOTTING=False
#     pass

#convert these which are list of tensors to just a tenosr now.
component_loss_train = torch.zeros(len(component_losses_train), len(component_losses_train[0]))
for i in range(len(component_losses_train)):
    component_loss_train[i, :] = component_losses_train[i]


component_loss_test = torch.zeros(len(component_losses_test), len(component_losses_test[0]))
for i in range(len(component_losses_test)):
    component_loss_test[i, :] = component_losses_test[i]

print(component_losses_test)


if DO_PLOTTING:
    print("Plotting...")
    from plotting import plotting_standard
    fields = [field[1] for field in yt.load(react_data.output_files[0]).field_list]

    plot_class = plotting_standard(model, fields, test_loader, cost_per_epoc, component_loss_test,
                component_loss_train, cost_per_epoc_test, output_dir)

    plot_class.do_all_plots()




if SAVE_MODEL:
    file_name = output_dir + 'my_model.pt'
    if os.path.exists(file_name):
        print("Overwritting file:", file_name)
        val = input("Overwrite file? y/n: ")
        if val == "y":
            torch.save(model.state_dict(), file_name)
            np.savetxt(output_dir + "/cost_per_epoch.txt", cost_per_epoc)
            np.savetxt(output_dir + "/component_losses_test.txt", component_loss_test)
            np.savetxt(output_dir + "/component_losses_train.txt", component_loss_train)

        else:
            pass
    else:
        torch.save(model.state_dict(), file_name)
        np.savetxt(output_dir + "/cost_per_epoch.txt", cost_per_epoc)
        np.savetxt(output_dir + "/component_losses_test.txt", component_loss_test)
        np.savetxt(output_dir + "/component_losses_train.txt", component_loss_train)
