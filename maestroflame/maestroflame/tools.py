import sys
import torch

class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
        self.log.write("MaestroFlame\n")
        self.log.close()

    def write(self, message):
        self.terminal.write(message + '\n')
        self.log = open(self.log_file, "a")
        self.log.write(message)
        self.log.close()

    def flush(self):
        pass

def scaling_func(x, f, barrier1, barrier2, scaling, smallest_num=torch.tensor([1e-30])):

    mask = x==0

    bump1 = f(barrier1) - f(scaling*torch.log(barrier1)) #makes left barrier continuous
    bump2 = f(barrier2) - f(scaling*torch.log(barrier2)) #makes right barrier continuous


    y1 = my_heaviside(x - barrier1, torch.tensor([0.]))
    y2 = -my_heaviside(x - barrier2, torch.tensor([0.])) + 1
    y3 = -my_heaviside(x - barrier1, torch.tensor([0.])) + 1
    y4 = my_heaviside(x - barrier2, torch.tensor([0.]))

    A = y1*y2
    B = torch.abs(y3-y4)
    C = y3
    D = y4

    #asserts everything is above zero.
    #we can't really know the smallest loss function value before we do ML.
    #But the default (1e-30) is defintley below it.
    bump3 = f(scaling*torch.log(smallest_num)+bump1)

    sol = (C*f(scaling*torch.log(x)+bump1) + f(x)*A + D*f(scaling*torch.log(x)+bump2))-bump3

    sol[mask] = 0.0

    return sol
