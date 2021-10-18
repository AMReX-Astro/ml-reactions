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
from .reactdataset import ReactDataset
from .losses import component_loss_f, component_loss_f_L1
from .plotting import plotting_standard, plotting_pinn


yt.funcs.mylog.setLevel(40) # Gets rid of all of the yt info text, only errors.
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation) #ignore plt depreciations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NuclearReactionML:


    def __init__(self, data_path, input_prefix, output_prefix, plotfile_prefix,
                output_dir, log_file, DEBUG_MODE=True, DO_PLOTTING=True,
                SAVE_MODEL=True, DO_HYPER_OPTIMIZATION=False):
                """
                data_path (string): this is the path to your data files

                input_prefix (string): This is the input prefix of your
                                       datafile names.

                output_prefix (string): This is the output prefix of your
                                        datafile names.

                plotfile_prefix (string): Plotfile prefixes, used for
                                          visualization purposes

                output_dir (string): By default, this package will save your model,
                                     logs of the training and testing data during
                                     training, and plots to a directory. Here
                                     you specify that directory. Must be a new
                                     directory or you will get an error to
                                     prevent the overwrite of data.

                log_file (string): The log file. Everything that is printed
                                   during training also goes into this file
                                   in case something gets interrupted.


                DEBUG_MODE (bool): This takes a small cut of the data (5 plotfiles)
                                   to train on. This is useful for debuggind since
                                   you won't have to deal with unwieldy amounts
                                   of data. Default=True

                DO_PLOTTING (bool): Whether to do the error plots or not.
                                    Saved in output_dir Default=True

                SAVE_MODEL (bool):  Whether to save the pytorch model file or
                                    not. Saved in output_dir Deafult=True

                DO_HYPER_OPTIMIZATION (bool): This still needs to be tweaked.
                                    As of right now it does hyperparameter
                                    optimization over an unfeasible amount of
                                    parameters. It does then however, take its
                                    results and automatically makes a pytorch
                                    network out of it to train. The results
                                    will be stored in the log_file Default=False
                """

                self.DO_PLOTTING = DO_PLOTTING
                self.output_dir = output_dir
                self.SAVE_MODEL = SAVE_MODEL
                self.DO_HYPER_OPTIMIZATION = DO_HYPER_OPTIMIZATION

                if os.path.isdir(output_dir) and (len(os.listdir(output_dir)) != 0):
                    print(f"Directory {output_dir} exists and is not empty.")
                    print("Please change output_dir or remove the directory to prevent overwritting data.")
                    sys.exit()

                isdir = os.path.isdir(output_dir)
                if not isdir:
                    os.mkdir(output_dir)

                from .tools import Logger
                self.logger = Logger(log_file)

                now = datetime.now()
                dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
                self.logger.write(f"Model starting on : {dt_string}")
                self.logger.write(f"input_prefix {input_prefix}")
                self.logger.write(f"output_prefix {output_prefix}")
                self.logger.write(f"output_dir {output_dir}")
                self.logger.write(f"DEBUG_MODE {DEBUG_MODE}")
                self.logger.write(f"DO_PLOTTING {DO_PLOTTING}")
                #self.logger.write(f"log_loss_option {log_loss_option}")
                self.logger.write(f"DO_HYPER_OPTIMIZATION {DO_HYPER_OPTIMIZATION}")


                #LOADING DATA----------------------------------------------------------
                # plotfiles = glob(data_path + plotfile_prefix)
                # plotfiles = sorted(plotfiles)
                # plotfiles = plotfiles[:-2] #cut after divuiter and initproj
                # plotfiles = [plotfiles[-1]] + plotfiles[:-1] #move initdata to front.
                #make_movie(plotfiles, movie_name='enuc.mp4', var='enuc')

                react_data = ReactDataset(data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=DEBUG_MODE)

                #Normalize density, temperature, and enuc
                dens_fac = torch.max(react_data.input_data[:, 14, :])
                temp_fac = torch.max(react_data.input_data[:, 15, :])
                enuc_fac = torch.max(react_data.output_data[:, 13, :])
                react_data.input_data[:, 14, :]  = react_data.input_data[:, 14, :]/dens_fac
                react_data.input_data[:, 15, :]  = react_data.input_data[:, 15, :]/temp_fac
                react_data.output_data[:, 13, :] = react_data.output_data[:, 13, :]/enuc_fac

                #save these factors to a file
                arr = np.array([dens_fac.item(), temp_fac.item(), enuc_fac.item()])
                np.savetxt(self.output_dir + 'scaling_factors.txt', arr, header='Density, Temperature, Enuc factors (ordered)')

                self.fields = [field for field in yt.load(react_data.output_files[0])._field_list]


                #percent cut for testing
                percent_test = 10
                N = len(react_data)

                Num_test  = int(N*percent_test/100)
                Num_train = N-Num_test

                train_set, test_set = torch.utils.data.random_split(react_data, [Num_train, Num_test])

                self.train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
                self.test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)


    def hyperparamter_optimization(self):
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
            hyper_results = do_h_opt(self.train_loader, self.test_loader, BATCHSIZE, CLASSES, EPOCHS,
                                  LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                                  LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                                  n_trials, timeout, in_features)

            print_h_opt_results(hyper_results)

            model = OC_Net(react_data.input_data.shape[1], react_data.output_data.shape[1], hyper_results)

            # Get model to cuda if possible
            model.to(device=device)

            optimizer = hyper_results['optimizer']
            lr = hyper_results['lr']
            if optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            elif optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=lr)
            else:
                self.logger.write("unsupported optimizer, please define it.")
                sys.exit()

    def train(self, model, optimizer, num_epochs, criterion, save_every_N=np.Inf):
            '''
            save_every_N - int representing every N epochs output the pytorch model.
                         Defaulted at infinity, meaning it won't output intermediately
            '''

            if save_every_N < np.Inf:
                os.mkdir(self.output_dir+'intermediate_output/')


            #As we test different loss functions, its important to keep a consistent one when
            #plotting or else we have no way of comparing them.
            criterion_plotting = nn.MSELoss()

            #plot storage
            self.cost_per_epoc = [] #stores total loss at each epoch
            self.component_losses_test = [] #stores component wise loss at each epoch (test data)
            self.component_losses_train = [] #stores component wise loss at each epoch (train data)
            self.cost_per_epoc_test = [] #stores total cost per epoc on testing data

            for epoch in range(num_epochs):
                losses = []
                plotting_losses = []

                for batch_idx, (data, targets) in enumerate(self.train_loader):
                    # Get data to cuda if possible
                    data = data.to(device=device)
                    targets = targets.to(device=device)

                    # forward
                    pred = model(data)
                    loss = criterion(pred, targets)

                    losses.append(loss.item())

                    if self.DO_PLOTTING:
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


                self.logger.write(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
                self.component_losses_train.append(component_loss/batch_idx)
                self.cost_per_epoc.append(sum(plotting_losses) / len(plotting_losses))

                with torch.no_grad():
                    if epoch % save_every_N == 0 and epoch != 0:
                        directory = self.output_dir+'intermediate_output/epoch'+str(epoch)+'/'
                        os.mkdir(directory)


                        torch.save(model.state_dict(), directory+'my_model.pt')
                        np.savetxt(directory + "/cost_per_epoch.txt", self.cost_per_epoc)
                        np.savetxt(directory + "/component_losses_test.txt", self.component_losses_test)
                        np.savetxt(directory + "/component_losses_train.txt", self.component_losses_train)


                        plot_class = plotting_standard(model, self.fields, self.test_loader, self.cost_per_epoc, np.array(self.component_losses_test),
                                                np.array(self.component_losses_train), self.cost_per_epoc_test, directory)

                        plot_class.do_all_plots()


                if self.DO_PLOTTING:
                    model.eval()

                    with torch.no_grad():
                        #Evaulate NN on testing data.
                        for batch_idx, (data, targets) in enumerate(self.test_loader):
                            # Get data to cuda if possible
                            data = data.to(device=device)
                            targets = targets.to(device=device)

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

                        self.cost_per_epoc_test.append(sum(losses) / len(losses))
                        self.component_losses_test.append(component_loss/batch_idx)

                    model.train()

            self.component_losses_test = np.array(self.component_losses_test)
            self.component_losses_train = np.array(self.component_losses_train)

            self.model = model

            if self.SAVE_MODEL:
                self.logger.write("Saving...")
                file_name = self.output_dir + 'my_model.pt'
                if os.path.exists(file_name):
                    self.logger.write("Overwritting file:", file_name)
                    os.rename(file_name, file_name+'.backup')

                torch.save(model.state_dict(), file_name)
                np.savetxt(self.output_dir + "/cost_per_epoch.txt", self.cost_per_epoc)
                np.savetxt(self.output_dir + "/component_losses_test.txt", self.component_losses_test)
                np.savetxt(self.output_dir + "/component_losses_train.txt", self.component_losses_train)




    def plot(self):
            self.logger.write("Plotting...")

            plot_class = plotting_standard(self.model, self.fields, self.test_loader, self.cost_per_epoc, self.component_losses_test,
                        self.component_losses_train, self.cost_per_epoc_test, self.output_dir)

            plot_class.do_all_plots()




class NuclearReactionPinn:


    def __init__(self, data_path, input_prefix, output_prefix, plotfile_prefix,
                output_dir, log_file, DEBUG_MODE=False, DO_PLOTTING=True,
                SAVE_MODEL=True, DO_HYPER_OPTIMIZATION=False, DO_GRADIENT_PLOT=False):
                """
                data_path (string): this is the path to your data files

                input_prefix (string): This is the input prefix of your
                                       datafile names.

                output_prefix (string): This is the output prefix of your
                                        datafile names.

                plotfile_prefix (string): Plotfile prefixes, used for
                                          visualization purposes

                output_dir (string): By default, this package will save your model,
                                     logs of the training and testing data during
                                     training, and plots to a directory. Here
                                     you specify that directory. Must be a new
                                     directory or you will get an error to
                                     prevent the overwrite of data.

                log_file (string): The log file. Everything that is printed
                                   during training also goes into this file
                                   in case something gets interrupted.


                DEBUG_MODE (bool): This takes a small cut of the data (5 plotfiles)
                                   to train on. This is useful for debuggind since
                                   you won't have to deal with unwieldy amounts
                                   of data. Default=True

                DO_PLOTTING (bool): Whether to do the error plots or not.
                                    Saved in output_dir Default=True

                SAVE_MODEL (bool):  Whether to save the pytorch model file or
                                    not. Saved in output_dir Deafult=True

                DO_HYPER_OPTIMIZATION (bool): TODO. Not yet implemented for PINNS
                                              Default=False

                DO_GRADIENT_PLOT (bool): A useful debugging feature to search for
                                         exploding/vanishing gradients. This plots
                                         gradient of loss with respect to all the
                                         neural network parameters on a graph.
                                         This is a different option then DO_PLOTTING
                                         because it takes quite a long time and
                                         is more for debugging. Default=False
                """



                self.DO_PLOTTING = DO_PLOTTING
                self.output_dir = output_dir
                self.DO_GRADIENT_PLOT = DO_GRADIENT_PLOT
                self.SAVE_MODEL = SAVE_MODEL
                self.DO_HYPER_OPTIMIZATION = DO_HYPER_OPTIMIZATION


                if DO_HYPER_OPTIMIZATION:
                    print("Hyperparameter optimization is not yet supported for pinns.")
                    sys.exit()

                if os.path.isdir(output_dir) and (len(os.listdir(output_dir)) != 0):
                    print(f"Directory {output_dir} exists and is not empty.")
                    print("Please change output_dir or remove the directory to prevent overwritting data.")
                    sys.exit()


                isdir = os.path.isdir(output_dir)
                if not isdir:
                    os.mkdir(output_dir)

                from .tools import Logger
                self.logger = Logger(log_file)


                now = datetime.now()
                dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
                self.logger.write(f"Model starting on : {dt_string}")
                self.logger.write(f"input_prefix {input_prefix}")
                self.logger.write(f"output_prefix {output_prefix}")
                self.logger.write(f"output_dir {output_dir}")
                self.logger.write(f"DEBUG_MODE {DEBUG_MODE}")
                self.logger.write(f"DO_PLOTTING {DO_PLOTTING}")
                self.logger.write(f"DO_HYPER_OPTIMIZATION {DO_HYPER_OPTIMIZATION}")


                plotfiles = glob(data_path + plotfile_prefix)
                plotfiles = sorted(plotfiles)
                plotfiles = plotfiles[:-2] #cut after divuiter and initproj
                plotfiles = [plotfiles[-1]] + plotfiles[:-1] #move initdata to front.


                react_data = ReactDataset(data_path, input_prefix, output_prefix, plotfile_prefix, DEBUG_MODE=DEBUG_MODE)
                self.nnuc = int(react_data.output_data.shape[1]/2 - 1)

                #Normalize density, temperature, and enuc
                dens_fac = torch.max(react_data.input_data[:, self.nnuc+1, :])
                temp_fac = torch.max(react_data.input_data[:, self.nnuc+2, :])
                enuc_fac = torch.max(react_data.output_data[:, self.nnuc, :])
                enuc_dot_fac = torch.max(react_data.output_data[:, 2*(self.nnuc+1) - 1, :])


                #RHS temperature at tn+1 (obtained from calling EOS)
                #rhs_fac = torch.max(react_data.output_data[:, 14, :])

                react_data.input_data[:, self.nnuc+1, :]  = react_data.input_data[:, self.nnuc+1, :]/dens_fac
                react_data.input_data[:, self.nnuc+2, :]  = react_data.input_data[:, self.nnuc+2, :]/temp_fac
                react_data.output_data[:, self.nnuc, :] = react_data.output_data[:, self.nnuc, :]/enuc_fac
                react_data.output_data[:, 2*(self.nnuc+1) - 1, :] = react_data.output_data[:, 2*(self.nnuc+1) - 1, :]/enuc_dot_fac

                #dpndx[enuc] = enuc_fac/enuc_dot_fac * dpndx[enuc]

                #percent cut for testing
                percent_test = 10
                N = len(react_data)

                Num_test  = int(N*percent_test/100)
                Num_train = N-Num_test

                train_set, test_set = torch.utils.data.random_split(react_data, [Num_train, Num_test])

                self.train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
                self.test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)

                self.fields = [field for field in yt.load(react_data.output_files[0])._field_list]
                input_fields = [field for field in yt.load(react_data.input_files[0])._field_list]
                self.input_fields = ['dt'] + input_fields


# if DO_HYPER_OPTIMIZATION:
#     BATCHSIZE = 16
#     CLASSES = react_data.output_data.shape[1]
#     in_features = react_data.input_data.shape[1]
#     EPOCHS = 10
#     LOG_INTERVAL = 10
#     N_TRAIN_EXAMPLES = BATCHSIZE * 30
#     N_VALID_EXAMPLES = BATCHSIZE * 10
#     #hyperparameter space
#     LAYERS = [1,5]
#     UNITS = [4, 128] #nodes per layer
#     DROPOUT_RATE = [0, 0.5]
#     LEARNING_RATE = [1e-7, 1e-1]
#     OPTIMIZERS = ["Adam", "RMSprop", "SGD"]
#     #optimizer study
#     if DEBUG_MODE:
#         n_trials = 100
#     else:
#         n_trials=100
#     timeout=600
#
#     from hyperparamter_optimization import do_h_opt
#     hyper_results = do_h_opt(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
#                           LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
#                           LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
#                           n_trials, timeout, in_features)
#
#     model = OC_Net(react_data.input_data.shape[1], react_data.output_data.shape[1]/2, hyper_results)
#
#
#     optimizer = hyper_results['optimizer']
#     lr = hyper_results['lr']
#     if optimizer == 'Adam':
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#     elif optimizer == 'RMSprop':
#         optimizer = optim.RMSprop(model.parameters(), lr=lr)
#     elif optimizer == 'SGD':
#         optimizer = optim.SGD(model.parameters(), lr=lr)
#     else:
#         self.logger.write("unsupported optimizer, please define it.")
#         sys.exit()




    def train(self, model, optimizer, num_epochs, criterion, save_every_N=np.Inf):
        '''
        save_every_N - int representing every N epochs output the pytorch model.
                     Defaulted at infinity, meaning it won't output intermediately
        '''

        if save_every_N < np.Inf:
            os.mkdir(self.output_dir+'intermediate_output/')


        #As we test different loss functions, its important to keep a consistent one when
        #plotting or else we have no way of comparing them.
        def criterion_plotting(prediction, targets):
            L = nn.MSELoss()
            return L(prediction, targets[:, :self.nnuc+1])

        #plot storage
        self.cost_per_epoc = [] #stores total loss at each epoch
        self.component_losses_test = [] #stores component wise loss at each epoch (test data)
        self.component_losses_train = [] #stores component wise loss at each epoch (train data)
        self.d_component_losses_test = [] #stores derivative component wise loss at each epoch (test data)
        self.d_component_losses_train = [] #stores derivative component wise loss at each epoch (train data)
        self.cost_per_epoc_test = []
        self.different_loss_metrics = [] #list of arrays of the various loss metrics defined in criterion

        if self.DO_GRADIENT_PLOT:
            #inputs plot
            plt.figure()
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                if batch_idx==0:
                    data_whole = data
                else:
                    data_whole = torch.cat((data_whole, data))
            N = data_whole.shape[0]
            for i in range(data_whole.shape[1]):
                plt.scatter([i]*N, data_whole[:, i], label=input_fields[i])
            plt.yscale("log")
            plt.legend()
            plt.savefig(self.output_dir + "input_fig.png",bbox_inches='tight')

            stride = 1#every how many epochs do we plot the grad?
            cmap = plt.get_cmap('gist_rainbow', num_epochs//stride)
            #cmap = cmap(np.linspace(1, num_epochs, num_epochs))
            cmap = cmap(np.linspace(1,num_epochs//stride, num_epochs//stride).astype(int).tolist())
            fig, ax_grad = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))



        labels = []
        xs = []
        #train network
        for epoch in range(num_epochs):
            losses = []
            plotting_losses = []
            diff_losses = []

            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Get data to cuda if possible
                data = data.to(device=device)
                data.requires_grad=True

                targets = targets.to(device=device)

                # forward
                prediction = model(data)

                # calculate derivatives
                dXdt = torch.zeros_like(prediction)
                for n in range(self.nnuc+1):
                    if data.grad is not None:
                        #sets componnents to zero if not already zero
                        data.grad.data.zero_()

                    prediction[:, n].backward(torch.ones_like(prediction[:,n]), retain_graph=True)
                    dXdt[:, n] = data.grad.clone()[:, 0]
                    data.grad.data.zero_()

                dXdt.requires_grad = True
                loss, array_loss = criterion(data, prediction, dXdt, targets)

                losses.append(loss.item())
                diff_losses.append(array_loss)

                # backward
                optimizer.zero_grad()
                loss.backward()

                #only do gradient plot for batch 0
                if self.DO_GRADIENT_PLOT:
                    with torch.no_grad():
                        if batch_idx == 0:
                            for i, (name, param) in enumerate(model.named_parameters()):
                                if epoch == 0:
                                    xs.append(i)
                                    labels.append(name)
                                if epoch % stride == 0:
                                    N=len(param.grad.flatten())
                                    if i==0:
                                        ax_grad[0].scatter([i]*N, np.abs(param.grad.flatten()), color =cmap[epoch//stride], s=2*num_epochs//stride-2*epoch//stride, label=f'epoch {epoch}')
                                        ax_grad[1].scatter([i]*N, np.abs(param.detach().numpy().flatten()), color =cmap[epoch//stride], s=2*num_epochs//stride-2*epoch//stride, label=f'epoch {epoch}')
                                    else:
                                        ax_grad[0].scatter([i]*N, np.abs(param.grad.flatten()), color =cmap[epoch//stride], s=2*num_epochs//stride-2*epoch//stride)
                                        ax_grad[1].scatter([i]*N, np.abs(param.detach().numpy().flatten()), color =cmap[epoch//stride], s=2*num_epochs//stride-2*epoch//stride)

                optimizer.step()

                #PLOTTING TERMS
                loss_plot = criterion_plotting(prediction, targets)
                plotting_losses.append(loss_plot.item())

                loss_c = component_loss_f(prediction, targets[:, :self.nnuc+1])
                loss_c = np.array(loss_c.tolist())

                #L1 loss bc big errors at first squaring big numbers results in nans
                dloss_c = component_loss_f_L1(dXdt, targets[:, self.nnuc+1:])
                dloss_c = np.array(dloss_c.tolist())

                if batch_idx == 0:
                    component_loss = loss_c
                    d_component_loss = dloss_c

                else:
                    component_loss = component_loss + loss_c
                    d_component_loss = d_component_loss + dloss_c


            self.logger.write(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")
            #Cost per epoc
            self.cost_per_epoc.append(sum(plotting_losses) / len(plotting_losses))

            #cost_per_epoc.append(sum(losses) / len(losses))
            self.component_losses_train.append(component_loss/batch_idx)
            self.d_component_losses_train.append(d_component_loss/batch_idx)

            diff_losses = np.array(diff_losses)
            self.different_loss_metrics.append(diff_losses.sum(axis=0)/len(losses))

            with torch.no_grad():
                if epoch % save_every_N == 0 and epoch != 0:
                    directory = self.output_dir+'intermediate_output/epoch'+str(epoch)+'/'
                    os.mkdir(directory)


                    torch.save(model.state_dict(), directory+'my_model.pt')
                    np.savetxt(directory + "/cost_per_epoch.txt", self.cost_per_epoc)
                    np.savetxt(directory + "/component_losses_test.txt", self.component_losses_test)
                    np.savetxt(directory + "/component_losses_train.txt", self.component_losses_train)
                    np.savetxt(self.output_dir + "/d_component_losses_test.txt", self.d_component_losses_test)
                    np.savetxt(self.output_dir + "/d_component_losses_train.txt", self.d_component_losses_train)


                    plot_class = plotting_pinn(self.model, self.fields, self.test_loader, self.cost_per_epoc,
                                np.array(self.component_losses_test), np.array(self.component_losses_train),
                                np.array(self.d_component_losses_test), np.array(self.d_component_losses_train),
                                self.cost_per_epoc_test, np.array(self.different_loss_metrics),
                                self.output_dir)

                    plot_class.do_all_plots()




            if self.DO_PLOTTING:
                model.eval()

                for batch_idx, (data, targets) in enumerate(self.train_loader):
                    # Get data to cuda if possible
                    data = data.to(device=device)
                    targets = targets.to(device=device)

                    # forward
                    data.requires_grad=True

                    prediction = model(data)
                    loss = criterion_plotting(prediction, targets)
                    losses.append(loss.item())

                    # calculate derivatives
                    dXdt = torch.zeros_like(prediction)
                    for n in range(self.nnuc+1):
                        if data.grad is not None:
                            #sets componnents to zero if not already zero
                            data.grad.data.zero_()

                        prediction[:, n].backward(torch.ones_like(prediction[:,n]), retain_graph=True)
                        dXdt[:, n] = data.grad.clone()[:, 0]
                        data.grad.data.zero_()


                    # -- Component and Deritivave component loss
                    loss_c = component_loss_f(prediction, targets[:, :self.nnuc+1])
                    loss_c = np.array(loss_c.tolist())

                    #L1 loss bc big errors at first squaring big numbers results in nans
                    dloss_c = component_loss_f_L1(dXdt, targets[:, self.nnuc+1:])
                    dloss_c = np.array(dloss_c.tolist())

                    if batch_idx == 0:
                        component_loss = loss_c
                        d_component_loss = dloss_c
                    else:
                        component_loss = component_loss + loss_c
                        d_component_loss = d_component_loss + dloss_c

                model.train()

                self.cost_per_epoc_test.append(sum(losses) / len(losses))
                self.component_losses_test.append(component_loss/batch_idx)
                self.d_component_losses_test.append(d_component_loss/batch_idx)


        self.component_losses_train = np.array(self.component_losses_train)
        self.component_losses_test = np.array(self.component_losses_test)
        self.d_component_losses_train = np.array(self.d_component_losses_train)
        self.d_component_losses_test = np.array(self.d_component_losses_test)
        self.different_loss_metrics = np.array(self.different_loss_metrics)


        if self.SAVE_MODEL:
            self.logger.write("Saving...")
            file_name = self.output_dir + 'my_model_pinn.pt'
            if os.path.exists(file_name):
                self.logger.write("Overwritting file:", file_name)
                os.rename(file_name, file_name+'.backup')

            torch.save(model.state_dict(), file_name)
            np.savetxt(self.output_dir + "/cost_per_epoch.txt", self.cost_per_epoc)
            np.savetxt(self.output_dir + "/component_losses_test.txt", self.component_losses_test)
            np.savetxt(self.output_dir + "/component_losses_train.txt", self.component_losses_train)
            np.savetxt(self.output_dir + "/d_component_losses_test.txt", self.d_component_losses_test)
            np.savetxt(self.output_dir + "/d_component_losses_train.txt", self.d_component_losses_train)



        if self.DO_GRADIENT_PLOT:
            with torch.no_grad():

                ax_grad[0].set_xticks(xs, minor=False)
                ax_grad[0].set_xticklabels(labels, fontdict=None, minor=False, rotation='vertical')
                ax_grad[0].set_ylabel("Abs(param.grad) with respect to loss")
                ax_grad[0].set_yscale("log")
                ax_grad[1].set_yscale("log")
                ax_grad[1].set_ylabel("Abs(param)")
                ax_grad[0].set_title("Derivative of NN parameters")
                ax_grad[1].set_title("NN parameters")
                fig.savefig(self.output_dir + "gradient_plot.pdf", bbox_inches='tight')

        self.model = model


    def plot(self):

        self.logger.write("Plotting...")

        plot_class = plotting_pinn(self.model, self.fields, self.test_loader, self.cost_per_epoc,
                    self.component_losses_test, self.component_losses_train,
                    self.d_component_losses_test, self.d_component_losses_train,
                    self.cost_per_epoc_test, self.different_loss_metrics,
                    self.output_dir)

        plot_class.do_all_plots()
