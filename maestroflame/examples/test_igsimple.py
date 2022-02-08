# Python script to train an ML model for ignition_simple network
import maestroflame
import os
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

# The log file. Everything that is printed during training also goes into this file in case something
# gets interrupted.
log_file = output_dir + "log.txt"

# Check to see if the output directory already exists
if os.path.exists(output_dir):
    print(f"Overwriting existing output directory: {output_dir}")
    os.rename(output_dir, output_dir[:-1]+'_old')


## MODEL SETUP

nrml = NuclearReactionML(data_path, input_prefix, output_prefix, plotfile_prefix,
                output_dir, log_file, DEBUG_MODE=True, DO_PLOTTING=True,
                SAVE_MODEL=True, DO_HYPER_OPTIMIZATION=False, LOG_MODE=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 50

def selectModel(model_id = 1):
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
    model.to(device=device)
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

optimizer = optim.Adam(model.parameters(), lr=1e-5)

nrml.train(model, optimizer, num_epochs, criterion)
    
# need to put model on cpu for plotting
model.to(device=torch.device("cpu"))

nrml.plot()


## SAVE MODEL

# convert to torch script
print("Saving model ...")
net_module = torch.jit.script(model)
net_module.save("ts_model.pt")
print(net_module.code)
