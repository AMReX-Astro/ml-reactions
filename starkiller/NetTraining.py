import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from TrainingHistory import TrainingHistory

class HiddenNet(nn.Module):
    def __init__(self, n_independent, n_dependent,
                 n_hidden, hidden_depth, activations):
        super(HiddenNet, self).__init__()
        
        self.activations = activations
        self.input_layer = nn.Linear(n_independent, n_hidden)
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_depth):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
        self.output_layer = nn.Linear(n_hidden, n_dependent)
        
    def forward(self, x):
        x = self.activations[0](self.input_layer(x))
        for i, h in enumerate(self.hidden_layers):
            x = self.activations[i+1](h(x))
        x = self.output_layer(x)
        return x

class NetTraining(object):
    def __init__(self, system, x, x_test, abs_tol=1.0e-6, rel_tol=1.0e-6, use_cuda=False):
        self.system = system
        self.x = x
        self.x_test = x_test
        self.use_cuda = use_cuda

        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        # get the truth solution as a function of x
        self.y = self.system.sol(self.x)

        # get the truth solution at the test points x_test
        self.y_test = self.system.sol(self.x_test)

        # get the analytic right-hand-side as a function of y(x)
        # f(x) = dy(x)/dx
        self.dydx = self.system.rhs(self.y)

        if self.use_cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
            self.x_test = self.x_test.cuda()
            self.y_test = self.y_test.cuda()
            self.dydx = self.dydx.cuda()

        # we will want to propagate gradients through y, dydx, and x
        # so make them PyTorch Variables
        self.x = Variable(self.x, requires_grad=True)
        self.y = Variable(self.y, requires_grad=True)
        self.dydx = Variable(self.dydx, requires_grad=True)

        # we will need to evaluate gradients w.r.t. x multiple
        # times so tell PyTorch to save the gradient variable in x.
        self.x.retain_grad()

        # keep track of the training history
        self.h = TrainingHistory()
        self.h.set_training_test_data(self.system,
                                      self.x.cpu().data.numpy(),
                                      self.y.cpu().data.numpy(),
                                      self.dydx.cpu().data.numpy(),
                                      self.x_test.cpu().data.numpy(),
                                      self.y_test.cpu().data.numpy())

    def init_net(self, n_i=1, n_d=1, n_h=1, depth_h=1, activations=None, optimizer_type="Adam"):
        self.n_independent = n_i
        self.n_dependent = n_d
        self.n_hidden = n_h
        self.depth_hidden = depth_h

        self.net = HiddenNet(n_independent=n_i, n_dependent=n_d,
                             n_hidden=n_h, hidden_depth=depth_h,
                             activations=activations)

        if self.use_cuda:
            self.net.cuda()

        if optimizer_type=="SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        elif optimizer_type=="Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        else:
            print("optimizer type not recognized")
            assert(optimizer_type=="SGD" or optimizer_type=="Adam")

        self.loss_func = torch.nn.MSELoss()

        print(self.net)

    def save_history(self, history_file="training_history.h5"):
        self.h.save_history(history_file)

    @staticmethod
    def mse_loss(input, target):
        return ((input - target)**2).sum() / input.data.nelement()

    def rms_weighted_error(self, input, target, solution):
        error_weight = self.abs_tol + self.rel_tol * torch.abs(solution)
        weighted_error = (input - target) / error_weight
        rms_weighted_error = torch.sqrt((weighted_error**2).sum() / input.data.nelement())
        return rms_weighted_error

    # this function is the training loop over epochs
    # where 1 epoch trains over the whole training dataset
    def train_error(self, NumEpochs, start_epoch=0):
        total_time = 0.0
        for t in range(NumEpochs):
            t = t + start_epoch
            
            net_start_time = time.time()

            # calculate prediction given the current net state
            prediction = self.net(self.x)

            # calculate error between prediction and analytic truth y
            #loss0 = torch.sqrt(mse_loss(prediction, y))
            loss0 = self.rms_weighted_error(prediction, self.y, self.y)

            # first, zero out the existing gradients to avoid
            # accumulating gradients on top of existing gradients
            self.net.zero_grad()

            # calculate gradients d(prediction)/d(x) for each component
            def get_component_gradient(n):
                if self.x.grad is not None:
                    self.x.grad.data.zero_()

                # now get the gradients dp_n/dt
                prediction[:,n].backward(torch.ones_like(prediction[:,n]), retain_graph=True)
                # clone the x gradient to save a copy of it as dp_n/dt
                dpndx = self.x.grad.clone()
                # clear the x gradient for the loss gradient below
                self.x.grad.data.zero_()
                
                # return dp_n/dt
                return dpndx
            
            dpdx = torch.ones_like(prediction)

            for i in range(self.n_dependent):
                dpdx[:,i] = torch.flatten(get_component_gradient(i))

            # define the error of the prediction derivative using the analytic derivative
            loss1 = torch.sqrt(self.loss_func(dpdx, self.dydx))
            #loss1 = rms_weighted_error(dpdx, dydx, dydx, abs_tol, rel_tol)

            # total error combines the error of the prediction (loss0) with 
            # the error of the prediction derivative (loss1)
            loss = loss0 + loss1

            # clear gradients for the next training iteration
            self.optimizer.zero_grad()

            # compute backpropagation gradients
            loss.backward()

            # apply gradients to update the weights
            self.optimizer.step()
            
            net_end_time = time.time()
            net_time = net_end_time - net_start_time
            total_time += net_time
            average_net_time = total_time / (t - start_epoch + 1.0)

            # log errors, check for early convergence
            if t % 100 == 0:
                # only calculate the following if we're doing I/O
                # get error with testing samples
                prediction_test = self.net(self.x_test)
                #test_loss = torch.sqrt(loss_func(prediction_test, self.y_test)).cpu().data.numpy()
                test_loss = self.rms_weighted_error(prediction_test, self.y_test, self.y_test)
                test_loss = test_loss.cpu().data.numpy()
            
                # evaluate the analytic right-hand-side function at the prediction value
                prhs = self.system.rhs(prediction)
                
                # log errors and epoch number
                self.h.epochs.append(t)
                self.h.losses.append(loss.cpu().data.numpy())
                self.h.losses0.append(loss0.cpu().data.numpy())
                self.h.losses1.append(loss1.cpu().data.numpy())
                self.h.test_losses.append(test_loss)

                # Save history for later analysis
                self.h.model_history.append(prediction.cpu().data.numpy())
                self.h.model_rhs_history.append(prhs.cpu().data.numpy())
                self.h.model_grad_history.append(dpdx.cpu().data.numpy())
                self.h.test_model_history.append(prediction_test.cpu().data.numpy())

                # Print epoch/error notifications
                print("epoch ", t, " with error: ", self.h.losses[-1],
                    "average time/epoch:", average_net_time)

            # Stop early if our errors are plateauing
            if t > 1000 and False:
                # do a quadratic polynomial fit and see if we will
                # need more than NumEpochs for the error e to vanish:
                # e / (d(e)/d(epoch)) > NumEpochs ?
                # if so, then break out of the training loop ...
                xfit = self.epochs[-4:]
                efit = self.losses[-4:]
                coef = np.polyfit(xfit, efit, 2)
                
                if coef[2]/coef[1] > NumEpochs:
                    break
