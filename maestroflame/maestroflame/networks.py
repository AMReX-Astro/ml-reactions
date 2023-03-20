import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, h1, h2, h3, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.CELU(alpha=100.0)
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.CELU(alpha=100.0)
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.CELU(alpha=1000.0)
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

#this is just a testing network to compare how tanh does vs celu
#We think there is a vanishing gradient problem.
class Net_tanh(nn.Module):
    def __init__(self, input_size, h1, h2, h3, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
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

# Net inspired by U-Net
class U_Net(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh()
        self.fc5 = nn.Linear(h4, output_size)
        
        # layers between non-consecutive layers
        self.io = nn.Linear(input_size, output_size)
        self.fc1to4 = nn.Linear(h1, h4)

    def forward(self, x):
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2))
        x4 = self.fc4(x3)
        x4 = self.ac4(x4 + self.fc1to4(x1))
        x5 = self.fc5(x4) + self.io(x)
        return x5

# Nets inspired by ResNet
class ResNet(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh()
        self.fc5 = nn.Linear(h4, output_size)
        
        # layers between non-consecutive layers 
        self.fc0to3 = nn.Linear(input_size, h3)
        self.fc2to5 = nn.Linear(h2, output_size)

    def forward(self, x):
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2) + self.fc0to3(x))
        x4 = self.ac4(self.fc4(x3))
        x5 = self.fc5(x4) + self.fc2to5(x2)
        return x5
    
class Cross_ResNet(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh()
        self.fc5 = nn.Linear(h4, output_size)
        
        # layers between non-consecutive layers
        self.fc1to4 = nn.Linear(h1, h4)
        self.fc2to5 = nn.Linear(h2, output_size)

    def forward(self, x):
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2))
        x4 = self.ac4(self.fc4(x3) + self.fc1to4(x1))
        x5 = self.fc5(x4) + self.fc2to5(x2)
        return x5

class Combine_Net3(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, output_size, relu=False):
        super().__init__()
        self.relu = relu
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh()
        self.fc5 = nn.Linear(h4, h5)
        self.ac5 = nn.Tanh()
        self.fc6 = nn.Linear(h5, h6)
        self.ac6 = nn.Tanh()
        self.fc7 = nn.Linear(h6, h7)
        self.ac7 = nn.Tanh()
        self.fc8 = nn.Linear(h7, h8)
        self.ac8 = nn.Tanh()
        self.fc9 = nn.Linear(h8, h9)
        self.ac9 = nn.Tanh()
        self.fc10 = nn.Linear(h9, h10)
        self.ac10 = nn.Tanh()
        self.fc11 = nn.Linear(h10, output_size)
        self.ac11 = nn.ReLU()
        
        # Resnet connections 
        self.fc1to4 = nn.Linear(h1, h4)
        self.fc3to6 = nn.Linear(h3, h6)
        self.fc5to8 = nn.Linear(h5, h8)
        self.fc7to10 = nn.Linear(h7, h10)
        
        # U net connections
        self.fc1to10 = nn.Linear(h1, h10)
        self.fcio = nn.Linear(input_size, output_size)

    def forward(self, x):
        x1 = self.ac1(self.fc1(x))
        x2 = self.ac2(self.fc2(x1))
        x3 = self.ac3(self.fc3(x2))
        x4 = self.ac4(self.fc4(x3) + self.fc1to4(x1))
        x5 = self.ac5(self.fc5(x4))
        x6 = self.ac6(self.fc6(x5) + self.fc3to6(x3))
        x7 = self.ac7(self.fc7(x6))
        x8 = self.ac8(self.fc8(x7) + self.fc5to8(x5))
        x9 = self.ac9(self.fc9(x8))
        x10 = self.ac10(self.fc10(x9) + self.fc7to10(x7) + self.fc1to10(x1))
        x11 = self.fc11(x10) + self.fcio(x)
        if self.relu:
            x11 = self.ac11(x11)
        return x11

class Deep_Net(nn.Module):
    def __init__(self, input_size, h1, h2, h3, h4, h5, h6, h7, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.Tanh()
        self.fc4 = nn.Linear(h3, h4)
        self.ac4 = nn.Tanh()
        self.fc5 = nn.Linear(h5, h6)
        self.ac5 = nn.Tanh()
        self.fc6 = nn.Linear(h6, h7)
        self.ac6 = nn.Tanh()
        self.fc7 = nn.Linear(h7, output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.fc4(x)
        x = self.ac4(x)
        x = self.fc5(x)
        x = self.ac5(x)
        x = self.fc6(x)
        x = self.ac6(x)
        x = self.fc7(x)
        return x


class OC_Net(nn.Module):

    #'Optuma Compatable Network' Takes the optuma dictonary and creates
    #a network based on that.
    def __init__(self, input_size, output_size, hyper_results):
        super().__init__()

        self.n_layers = hyper_results['n_layers']

        n_units = [input_size]
        p_dropouts = []
        for i in range(self.n_layers):
            n_units.append(hyper_results["n_units_l{}".format(i)])
            p_dropouts.append(hyper_results["dropout_l{}".format(i)])
        n_units.append(output_size)


        self.layers = []
        self.dropouts = []
        self.activations = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(n_units[i], n_units[i+1]))
            self.dropouts.append(nn.Dropout(p_dropouts[i]))
            self.activations.append(nn.Tanh())

        #output layer
        self.layers.append(nn.Linear(n_units[self.n_layers], n_units[self.n_layers+1]))

        self.layers = nn.ModuleList(self.layers)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.activations = nn.ModuleList(self.activations)

    def forward(self, x):

        for i in range(self.n_layers):
            x = self.dropouts[i](self.activations[i](self.layers[i](x)))
        x = self.layers[self.n_layers](x)

        return x
