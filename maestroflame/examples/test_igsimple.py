# Python script to train an ML model for ignition_simple network
import maestroflame
import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim

from maestroflame.train import NuclearReactionML
from maestroflame.networks import *
from maestroflame.losses import log_loss, logX_loss, loss_mass_fraction_half_L

## FILE SETUP

# this is the path to your data files
data_path = '../data/flame_simple_2spec_steady_perturb/'

# These is the input/output prefix of your datafile names.
input_prefix = 'react_inputs_*'
output_prefix = 'react_outputs_*'

# Plotfile prefixes, used for visualization purposes.
plotfile_prefix = 'flame_*'

# By default, this package will save your model, logs of the training and testing data during training,
# and plots to a directory. Here you specify that directory.
output_dir = 'testing123/'

# Check to see if the output directory already exists
if os.path.exists(output_dir):
    hash = random.getrandbits(32)
    new_output_dir = output_dir[:-1] + f"_{hex(hash)}/"
    print(f"Replacing output directory {output_dir} with {new_output_dir}")
    #os.rename(output_dir, new_output_dir)
    output_dir = new_output_dir

# The log file. Everything that is printed during training also goes into this file in case something
# gets interrupted.
log_file = output_dir + "log.txt"


## MODEL SETUP

nrml = NuclearReactionML(data_path, input_prefix, output_prefix, plotfile_prefix,
                output_dir, log_file, DEBUG_MODE=True, DO_PLOTTING=True,
                SAVE_MODEL=True, DO_HYPER_OPTIMIZATION=False, LOG_MODE=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(f"Using {torch.cuda.device_count()} GPUs!")

num_epochs = 50

def selectModel(model_id = 1, device_opt = device):
    if model_id == 1:
        model = Net_tanh(4, 16, 16, 16, 3)
    elif model_id == 2:
        model = U_Net(4, 32, 8, 8, 16, 3)
    elif model_id == 3:
        model = ResNet(4, 16, 16, 16, 16, 3)
    elif model_id == 4:
        model = Cross_ResNet(4, 16, 16, 16, 16, 3)
    elif model_id == 5:
        # model = Deep_Net(4, 12, 12, 12, 12, 12, 12, 12, 3)
        model = Combine_Net3(4, 16, 8, 8, 8, 8, 8, 8, 8, 8, 16, 3)
    else:
        model = Net(4, 16, 16, 16, 3)

    # get model to cuda if possible
    # use all available GPUs
    if torch.cuda.device_count() > 1 and device_opt != torch.device('cpu'):
        model = nn.DataParallel(model)
    model.to(device_opt)
    return model

def loadModel(model_id, model_path):
    if model_id == 1:
        model = Net_tanh(4, 16, 16, 16, 3)
    elif model_id == 2:
        model = U_Net(4, 32, 8, 8, 16, 3)
    elif model_id == 3:
        model = ResNet(4, 16, 16, 16, 16, 3)
    elif model_id == 4:
        model = Cross_ResNet(4, 16, 16, 16, 16, 3)
    elif model_id == 5:
        # model = Deep_Net(4, 12, 12, 12, 12, 12, 12, 12, 3)
        model = Combine_Net3(4, 16, 8, 8, 8, 8, 8, 8, 8, 8, 16, 3)
    else:
        model = Net(4, 16, 16, 16, 3)

    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model.module.load_state_dict(torch.load(model_path))
    print(model)

    # get model to cuda if possible
    # use all available GPUs
    if torch.cuda.device_count() > 1 and device != torch.device('cpu'):
        model = nn.DataParallel(model)
    model.to(device)
    return model

def criterion(pred, target): 
    #loss1 = logX_loss(pred, target, nnuc=2)
    #loss2 = 10*loss_mass_fraction_half_L(pred, nnuc=2)
    #loss3 = loss_enuc(pred, target, mion)
    
    L = nn.MSELoss()
    F = nn.L1Loss()
    return L(pred[:, :3], target[:, :3]) + F(torch.sign(pred[:,2]), torch.sign(target[:,2]))
    #return loss1 #+ loss2 #+ loss3

    
## TRAIN MODEL

model_id = 5
model = selectModel(model_id)
print(f"Model {model_id} \n")
# model = loadModel(model_id, model_path="testing123_1416212/my_model.pt")

optimizer = optim.Adam(model.parameters(), lr=1e-5)

nrml.train(model, optimizer, num_epochs, criterion)

nrml.plot()


## SAVE MODEL

device_cpu = torch.device('cpu')

# reload model onto cpu
model_cpu = selectModel(model_id, device_cpu)
print("Loading model onto CPU...")

try:
    model_cpu.load_state_dict(torch.load(output_dir + "my_model.pt", map_location=device_cpu))
except RuntimeError:
    model_cpu.module.load_state_dict(torch.load(output_dir + "my_model.pt", map_location=device_cpu))
print(model_cpu)

# convert to torch script

print("Saving model...")
net_module = torch.jit.script(model_cpu)
net_module.save(output_dir + "ts_model.pt")
print(net_module.code)
