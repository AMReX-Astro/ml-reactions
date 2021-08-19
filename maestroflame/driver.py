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
DO_HYPER_OPTIMIZATION = False


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
output_dir = 'testing_output_scaling/'
log_file = output_dir + "log.txt"

if os.path.isdir(output_dir) and (len(os.listdir(output_dir)) != 0):
    print(f"Directory {output_dir} exists and is not empty.")
    print("Please change output_dir or remove the directory to prevent overwritting data.")
    sys.exit()


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

#save these factors to a file
arr = np.array([dens_fac.item(), temp_fac.item(), enuc_fac.item()])
np.savetxt(output_dir + 'scaling_factors.txt', arr, header='Density, Temperature, Enuc factors (ordered)')

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
from networks import Net, OC_Net, Deep_Net


if DEBUG_MODE:
    num_epochs = 1
else:
    num_epochs = 100


if DO_HYPER_OPTIMIZATION:
    BATCHSIZE = 16
    CLASSES = react_data.output_data.shape[1]
    in_features = react_data.input_data.shape[1]
    EPOCHS = 25
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
        n_trials = 100
    else:
        n_trials=1000
    timeout=1000

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

    #model = Net(react_data.input_data.shape[1], 64, 128, 64, react_data.output_data.shape[1])
    model = Deep_Net(react_data.input_data.shape[1], 32, 32, 32, 32, 32, 32, 32, react_data.output_data.shape[1])

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

for epoch in range(num_epochs):
    losses = []
    plotting_losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        pred = model(data)
        loss = criterion(pred, targets)

        losses.append(loss.item())

        if DO_PLOTTING:
            with torch.no_grad():
                loss_plot = criterion_plotting(pred, targets)
                plotting_losses.append(loss_plot.item())

                loss_c = component_loss_f(pred, targets)
                loss_c = np.array(loss_c.tolist())
                if batch_idx == 0:
                    component_loss = loss_c
                else:
                    component_loss = component_loss + loss_c

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()



    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
    component_losses_train.append(component_loss/batch_idx)
    cost_per_epoc.append(sum(plotting_losses) / len(plotting_losses))

    if DO_PLOTTING:

        #Evaulate NN on testing data.
        for batch_idx, (data, targets) in enumerate(test_loader):
            # forward
            pred = model(data)
            loss = criterion_plotting(pred, targets)
            losses.append(loss.item())

            loss_c = component_loss_f(pred, targets)
            loss_c = np.array(loss_c.tolist())
            if batch_idx == 0:
                component_loss = loss_c
            else:
                component_loss = component_loss + loss_c

        cost_per_epoc_test.append(sum(losses) / len(losses))
        component_losses_test.append(component_loss/batch_idx)


component_losses_test = np.array(component_losses_test)
component_losses_train = np.array(component_losses_train)



if SAVE_MODEL:
    print("Saving...")
    file_name = output_dir + 'my_model.pt'
    if os.path.exists(file_name):
        print("Overwritting file:", file_name)
        os.rename(file_name, file_name+'.backup')

    torch.save(model.state_dict(), file_name)
    np.savetxt(output_dir + "/cost_per_epoch.txt", cost_per_epoc)
    np.savetxt(output_dir + "/component_losses_test.txt", component_losses_test)
    np.savetxt(output_dir + "/component_losses_train.txt", component_losses_train)



if DO_PLOTTING:
    print("Plotting...")
    from plotting import plotting_standard
    fields = [field[1] for field in yt.load(react_data.output_files[0])._field_list]

    plot_class = plotting_standard(model, fields, test_loader, cost_per_epoc, component_losses_test,
                component_losses_train, cost_per_epoc_test, output_dir)

    plot_class.do_all_plots()


print("Success! :) \n")
sys.stdout.log.close()
