import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, h1, h2, h3, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.ac1 = nn.CELU()
        self.fc2 = nn.Linear(h1, h2)
        self.ac2 = nn.CELU()
        self.fc3 = nn.Linear(h2, h3)
        self.ac3 = nn.CELU()
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
