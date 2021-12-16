import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import sys
import julia
import time
from numpy import errstate,isneginf, isposinf

def make_movie(file_list, save_dir='./', var='enuc', movie_name="flame.mp4"):
    i = 1
    for file in file_list:
        ds = yt.load(file)
        sl = yt.SlicePlot(ds,2,var)
        sl.save("movie_imag{}.png".format(str(i).zfill(4)))
        i+=1
    os.system("ffmpeg -r 60 -pattern_type glob -i 'movie_imag*.png' -vcodec mpeg4 -y {}".format(movie_name))
    os.system("rm movie_imag*")
    Video("movie.mp4", embed=True)

class plotting_standard:
    #class to make it easy to plot things from driver. Set up all the data
    #then just call the methods for whatever plots you want.

    def __init__(self, model, fields, test_loader, cost_per_epoc,
                component_losses_test, component_losses_train,
                cost_per_epoc_test, output_dir):

        self.model = model
        self.fields = fields
        self.nnuc = len(fields)-1
        self.N_fields = len(fields)
        self.test_loader = test_loader
        self.cost_per_epoc = cost_per_epoc
        self.component_losses_test = component_losses_test
        self.component_losses_train = component_losses_train
        self.cost_per_epoc_test = cost_per_epoc_test
        self.output_dir = output_dir

        isdir = os.path.isdir(output_dir)
        if not isdir:
            os.mkdir(output_dir)


    def do_prediction_vs_solution_plot(self, use_julia=False):
        ############ Prediction Vs Solution Plot Should fall one y=x line.

        plt.figure()
        #N = react_data.output_data.shape[1]
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.nnuc+1))
        #fields = [field[1] for field in yt.load(react_data.output_files[0]).field_list]
        self.model.eval()

        with torch.no_grad():
            losses = []

            for batch_idx, (data, targets) in enumerate(self.test_loader):
            #pulling this data out and storing it then calling matplotlib as
            #few times as possible is much faster.
                if batch_idx==0:
                    data_whole = data
                    targets_whole = targets
                else:
                    data_whole = torch.cat((data_whole, data))
                    targets_whole = torch.cat((targets_whole, targets))

            #for batch_idx, (data, targets) in enumerate(self.test_loader):
            pred = self.model(data_whole)


            if use_julia:

                outer_start_time = time.time()
                start_time = time.time()


                # if self.output_dir[0] != '/':
                #     print("You must supply output_dir as an absolute path if you are going to \
                #             to use julia plotting!")
                #     sys.exit()
                #
                dir_path = os.path.dirname(os.path.realpath(__file__)) #+ '/maestroflame/'


                np.savetxt(self.output_dir + '/targets.txt', targets_whole)
                np.savetxt(self.output_dir + '/pred.txt', pred)
                #np.savetxt(self.output_dir + '/labels.txt', self.fields + [self.output_dir, dir_path], fmt="%s")

                from julia.api import LibJulia
                api = LibJulia.load()
                #api.sysimage = "sys_plots.so"
                api.sysimage = "../tools/sys.so"
                api.init_julia()
                from julia import Plots

                print("--- Internal Timing Julia")
                print("%s seconds to start julia " % (time.time() - start_time))
                start_time = time.time()

                #TODO: can't get labels to work.
                #Plots.scatter(pred, targets_whole, labels=self.fields, dpi=600)

                pred_plot = pred.cpu().detach().numpy()
                targets_plot = targets_whole.cpu().detach().numpy()

                Plots.scatter(pred_plot, targets_plot, dpi=600, xlabel='Prediction', ylabel='Solution')
                Plots.savefig(self.output_dir + "julia_prediction_vs_solution.png")

                #screen out zeros when taking log. Julia log plots can't handle zeros
                #so we just do it for them.

                with errstate(divide='ignore'):
                    pred_plot = np.log10(pred_plot)
                    targets_plot = np.log10(targets_plot)

                    #screen
                pred_plot_sc = pred_plot
                targets_plot_sc = targets_plot

                mask = np.isinf(pred_plot) + np.isinf(targets_plot) + np.isnan(pred_plot) + np.isnan(targets_plot)

                pred_plot_sc[mask]=0.
                targets_plot_sc[mask]=0.
                print("%s seconds to do preprocessing " % (time.time() - start_time))
                start_time = time.time()

                Plots.scatter(pred_plot, targets_plot, dpi=600, xlabel='log10(Prediction)', ylabel='log10(Solution)')
                Plots.savefig(self.output_dir + "julia_prediction_vs_solution_log.png")
                print("%s seconds to do plot " % (time.time() - start_time))
                print("%s seconds total for first julia plotting method " % (time.time() - outer_start_time))


                outer_start_time = time.time()
                start_time = time.time()

                if not os.path.isdir(dir_path + "/python_to_julia_data"):
                    os.mkdir(dir_path + "/python_to_julia_data")
                np.savetxt(dir_path + "/python_to_julia_data/"+"pred.txt", pred)
                np.savetxt(dir_path + "/python_to_julia_data/"+"targets.txt", targets_whole)
                str_to_julia = self.fields + [self.output_dir]

                np.savetxt(dir_path + "/python_to_julia_data/"+"labels.txt", str_to_julia, fmt="%s")
                print("%s seconds to save data to file " % (time.time() - start_time))
                start_time = time.time()


                os.system(f"julia {dir_path}"+"/pred_v_sol.jl")
                print("%s seconds to run julia pred_v_sol.jl " % (time.time() - start_time))
                print("%s seconds total to run julia method 2 " % (time.time() - outer_start_time))

            else:
                for i in range(pred.shape[1]):
                    plt.scatter(pred[:, i], targets_whole[:, i], color=colors[i], label=self.fields[i])
                plt.plot(np.linspace(0, 1), np.linspace(0,1), '--', color='orange')
                #plt.legend(yt.load(react_data.output_files[0]).field_list, colors=colors)
                plt.legend(bbox_to_anchor=(1, 1))
                plt.xlabel('Prediction')
                plt.ylabel('Solution')
                plt.savefig(self.output_dir + "/prediction_vs_solution.png", bbox_inches='tight')

                plt.yscale("log")
                plt.xscale("log")
                plt.savefig(self.output_dir + "/prediction_vs_solution_log.png", bbox_inches='tight')

        self.model.train()




    def do_cost_per_epoch_plot(self):
        ############## Cost per eppoc plot#####################
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)
        epocs = np.linspace(1, len(self.cost_per_epoc), num=len(self.cost_per_epoc))

        axs[0].plot(epocs, self.cost_per_epoc, label='Training Data')
        axs[0].plot(epocs, self.cost_per_epoc_test, label='Testing Data')

        axs[1].semilogy(epocs, self.cost_per_epoc)
        axs[1].semilogy(epocs, self.cost_per_epoc_test)

        fig.suptitle('Overall cost of training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        axs[0].legend(bbox_to_anchor=(1, 1))
        fig.savefig(self.output_dir + "/cost_vs_epoch.png", bbox_inches='tight')


    def do_component_loss_train_plot(self):

        #Component losses  train
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_train.shape[0]
        for i in range(self.component_losses_train.shape[1]):
            axs[0].plot(np.linspace(1, N, num=N),
                        self.component_losses_train[:, i], label=self.fields[i])

        for i in range(self.component_losses_train.shape[1]):
            axs[1].semilogy(np.linspace(1, N, num=N),
                            self.component_losses_train[:, i], label=self.fields[i])

        fig.suptitle('Component wise error in training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2))
        fig.savefig(self.output_dir + "/component_training_loss.png", bbox_inches='tight')



    def do_component_loss_test_plot(self):
        #Component losses  test
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_test.shape[0]
        for i in range(self.component_losses_test.shape[1]):
            axs[0].plot(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                        label=self.fields[i])

        for i in range(self.component_losses_test.shape[1]):
            axs[1].semilogy(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                            label=self.fields[i])

        fig.suptitle('Component wise error in testing data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2))
        fig.savefig(self.output_dir + "/component_testing_loss.png", bbox_inches='tight')

    def do_all_plots(self):
        self.do_cost_per_epoch_plot()
        self.do_component_loss_train_plot()
        self.do_component_loss_test_plot()
        self.do_prediction_vs_solution_plot()



class plotting_pinn:
    #class to make it easy to plot things from driver. Set up all the data
    #then just call the methods for whatever plots you want.

    def __init__(self, model, fields, test_loader, cost_per_epoc,
                component_losses_test, component_losses_train,
                d_component_losses_test, d_component_losses_train,
                cost_per_epoc_test, different_loss_metrics, output_dir):

        self.model = model
        self.fields = fields
        self.nnuc = int(len(fields)/2 - 1)
        self.N_fields = len(fields)
        self.test_loader = test_loader
        self.cost_per_epoc = cost_per_epoc
        self.component_losses_test = component_losses_test
        self.component_losses_train = component_losses_train
        self.d_component_losses_test = d_component_losses_test
        self.d_component_losses_train = d_component_losses_train
        self.cost_per_epoc_test = cost_per_epoc_test
        self.different_loss_metrics = different_loss_metrics
        self.output_dir = output_dir

        isdir = os.path.isdir(output_dir)
        if not isdir:
            os.mkdir(output_dir)


    def do_prediction_vs_solution_plot(self):
        ############ Prediction Vs Solution Plot Should fall one y=x line.
        plt.figure()
        #N = react_data.output_data.shape[1]
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, self.nnuc+1))
        #fields = [field[1] for field in yt.load(react_data.output_files[0]).field_list]
        self.model.eval()

        with torch.no_grad():
            losses = []

            for batch_idx, (data, targets) in enumerate(self.test_loader):
            #pulling this data out and storing it then calling matplotlib as
            #few times as possible is much faster.
                if batch_idx==0:
                    data_whole = data
                    targets_whole = targets
                else:
                    data_whole = torch.cat((data_whole, data))
                    targets_whole = torch.cat((targets_whole, targets))

            #for batch_idx, (data, targets) in enumerate(self.test_loader):
            pred = self.model(data_whole)

            for i in range(self.nnuc+1):
                plt.scatter(pred[:, i], targets_whole[:, i], color=colors[i], label=self.fields[i])

        self.model.train()

        plt.plot(np.linspace(0, 1), np.linspace(0,1), '--', color='orange')
        #plt.legend(yt.load(react_data.output_files[0]).field_list, colors=colors)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlabel('Prediction')
        plt.ylabel('Solution')
        plt.savefig(self.output_dir + "/prediction_vs_solution.png", bbox_inches='tight')

        plt.yscale("log")
        plt.xscale("log")
        plt.savefig(self.output_dir + "/prediction_vs_solution_log.png", bbox_inches='tight')


    def do_cost_per_epoch_plot(self):
        ############## Cost per eppoc plot#####################
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)
        epocs = np.linspace(1, len(self.cost_per_epoc), num=len(self.cost_per_epoc))

        axs[0].plot(epocs, self.cost_per_epoc, label='Training Data')
        axs[0].plot(epocs, self.cost_per_epoc_test, label='Testing Data')

        axs[1].semilogy(epocs, self.cost_per_epoc)
        axs[1].semilogy(epocs, self.cost_per_epoc_test)

        fig.suptitle('Overall cost of training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        axs[0].legend(bbox_to_anchor=(1, 1))
        fig.savefig(self.output_dir + "/cost_vs_epoch.png", bbox_inches='tight')


    def do_component_loss_train_plot(self):

        #Component losses  train
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_train.shape[0]
        for i in range(self.nnuc+1):
            axs[0].plot(np.linspace(1, N, num=N),
                        self.component_losses_train[:, i], label=self.fields[i])

        for i in range(self.nnuc+1):
            axs[1].semilogy(np.linspace(1, N, num=N),
                            self.component_losses_train[:, i], label=self.fields[i])

        fig.suptitle('Component wise error in training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2))
        fig.savefig(self.output_dir + "/component_training_loss.png", bbox_inches='tight')



    def do_component_loss_test_plot(self):
        #Component losses  test
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_test.shape[0]
        for i in range(self.nnuc+1):
            axs[0].plot(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                        label=self.fields[i])

        for i in range(self.nnuc+1):
            axs[1].semilogy(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                            label=self.fields[i])

        fig.suptitle('Component wise error in testing data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2), borderaxespad=0.)
        fig.savefig(self.output_dir + "/component_testing_loss.png", bbox_inches='tight')


    def do_dcomponent_loss_train_plot(self):

        #Component losses  train
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.d_component_losses_train.shape[0]
        for i in range(self.nnuc+1):
            axs[0].plot(np.linspace(1, N, num=N),
                        self.d_component_losses_train[:, i], label=self.fields[i])

        for i in range(self.nnuc+1):
            axs[1].semilogy(np.linspace(1, N, num=N),
                            self.d_component_losses_train[:, i], label=self.fields[i])

        fig.suptitle('Derivative component wise error in training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2), borderaxespad=0.)
        fig.savefig(self.output_dir + "/d_component_training_loss.png", bbox_inches='tight')


    def do_dcomponent_loss_test_plot(self):
        #Component losses  test
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.d_component_losses_test.shape[0]
        for i in range(self.nnuc+1):
            axs[0].plot(np.linspace(1, N, num=N), self.d_component_losses_test[:, i],
                        label=self.fields[i])

        for i in range(self.nnuc+1):
            axs[1].semilogy(np.linspace(1, N, num=N), self.d_component_losses_test[:, i],
                            label=self.fields[i])

        fig.suptitle('Derivative component wise error in testing data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2), borderaxespad=0.)
        fig.savefig(self.output_dir + "/d_component_testing_loss.png", bbox_inches='tight')


    def do_d_component_loss_test_plot(self):
        #Component derivative losses  test
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_test.shape[0]
        for i in range(self.nnuc+1, (self.nnuc+1)*2):
            axs[0].plot(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                        label=self.fields[i])

        for i in range(self.nnuc+1, (self.nnuc+1)*2):
            axs[1].semilogy(np.linspace(1, N, num=N), self.component_losses_test[:, i],
                            label=self.fields[i])

        fig.suptitle('Component wise error in testing data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2), borderaxespad=0.)
        fig.savefig(self.output_dir + "/d_component_testing_loss.png", bbox_inches='tight')



    def do_d_component_loss_train_plot(self):

        #Component losses  train
        fig = plt.figure()
        gs = fig.add_gridspec(2,1, hspace=0)
        axs = gs.subplots(sharex=True)

        N = self.component_losses_train.shape[0]
        for i in range(self.nnuc+1, (self.nnuc+1)*2):
            axs[0].plot(np.linspace(1, N, num=N),
                        self.component_losses_train[:, i], label=self.fields[i])

        for i in range(self.nnuc+1, (self.nnuc+1)*2):
            axs[1].semilogy(np.linspace(1, N, num=N),
                            self.component_losses_train[:, i], label=self.fields[i])

        fig.suptitle('Component wise error in training data')
        axs[1].set_xlabel("Num Epochs")
        axs[1].set_ylabel('Log Cost')
        axs[0].set_ylabel('Cost (MSE)')
        # Hide x labels and tick labels for all but bottom plot.
        for ax in axs:
            ax.label_outer()
        plt.legend(bbox_to_anchor=(1, 2), borderaxespad=0.)
        fig.savefig(self.output_dir + "/d_component_training_loss.png", bbox_inches='tight')


    def do_different_loss_plot(self):
        fig, ax = plt.subplots()


        epochs = np.linspace(1, len(self.cost_per_epoc), num=len(self.cost_per_epoc))

        for i in range(self.different_loss_metrics.shape[1]):

            sum = np.zeros_like(self.different_loss_metrics[:, 0])

            for j in range(i):
                sum += self.different_loss_metrics[:, i]


            plt.plot(epochs, self.different_loss_metrics[:, i] + sum, label='loss{}'.format(i+1))
            if i == 0:
                ax.fill_between(epochs, self.different_loss_metrics[:, i] + sum,  alpha=0.2)
            else:
                ax.fill_between(epochs, prev, self.different_loss_metrics[:, i] + sum,  alpha=0.2)

            prev = self.different_loss_metrics[:, i] + sum
        plt.legend(loc='upper left')
        plt.xlabel("epochs")
        plt.title("Various loss function values")
        fig.savefig(self.output_dir + "/different_loss_functions.png", bbox_inches='tight')

        plt.yscale("log")
        plt.title("Log Various loss function values")
        fig.savefig(self.output_dir + "/different_loss_log_functions.png", bbox_inches='tight')

    def do_all_plots(self):
        self.do_cost_per_epoch_plot()
        self.do_component_loss_train_plot()
        self.do_component_loss_test_plot()
        self.do_dcomponent_loss_train_plot()
        self.do_dcomponent_loss_test_plot()
        self.do_different_loss_plot()
        self.do_prediction_vs_solution_plot()
