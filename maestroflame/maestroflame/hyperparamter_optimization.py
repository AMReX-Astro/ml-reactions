import optuna
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np

def do_h_opt(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
                      LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                      LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                      n_trials, timeout, in_features):

    DEVICE = torch.device("cpu")
    DIR = os.getcwd()

    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", LAYERS[0], LAYERS[1])

        layers = []
        in_featurez = in_features # i have no idea why i had to do this
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), UNITS[0], UNITS[1])
            layers.append(nn.Linear(in_featurez, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), DROPOUT_RATE[0], DROPOUT_RATE[1])
            layers.append(nn.Dropout(p))

            in_featurez = out_features
        layers.append(nn.Linear(in_featurez, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def objective(trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZERS)
        lr = trial.suggest_float("lr", LEARNING_RATE[0], LEARNING_RATE[1], log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)



        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break


                optimizer.zero_grad()
                output = model(data)
                L = nn.MSELoss()
                loss = L(output, target)
                loss.backward()
                optimizer.step()


            #Validation of model
            model.eval()
            #we save the largest MSE of the set as the accuracy of the trial.
            accuracy = -np.inf
            L = nn.MSELoss()
            with torch.no_grad():
                 for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                        output = model(data)


                        local_accuracy = L(output, target)

                        if local_accuracy > accuracy:
                            accuracy = local_accuracy

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study.best_params


def do_h_opt_pinn(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
                      LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                      LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                      n_trials, timeout, in_features, nnuc):

    DEVICE = torch.device("cpu")
    DIR = os.getcwd()

    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", LAYERS[0], LAYERS[1])

        layers = []
        in_featurez = in_features # i have no idea why i had to do this
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), UNITS[0], UNITS[1])
            layers.append(nn.Linear(in_featurez, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), DROPOUT_RATE[0], DROPOUT_RATE[1])
            layers.append(nn.Dropout(p))

            in_featurez = out_features
        layers.append(nn.Linear(in_featurez, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)
    # Loss and optimizer
    def rms_weighted_error(input, target, solution, atol=1e-6, rtol=1e-6):
        error_weight = atol + rtol * torch.abs(solution)
        weighted_error = (input - target) / error_weight
        rms_weighted_error = torch.sqrt((weighted_error**2).sum() / input.data.nelement())
        return rms_weighted_error

    def loss_pinn(input, pred, target):



        #input mass fractions/density/temp at t=n
        dt     = input[:, 0]
        X_n    = input[:, 1:nnuc+1]
        rho_n  = input[:, nnuc+1]
        temp_n = input[:, nnuc+2]

        #output mass fractions (NN) at t=n+1
        X_nn_1    = pred[:, :nnuc]
        enuc_nn_1 = pred[:, nnuc]
        #rhs_nn_1  = pred

        #truth mass fractions (calculated) at t=n+1
        X_t_1    = target[:, :nnuc]
        enuc_t_1 = target[:, nnuc]
        #Asssume rho is constant, call eos to get updated T. then call rhs on updated variables.
        #This will all be done within maestro and the output is just loaded here.

        #rhs_t_1 = target[:, 15:28]

        dpndx = torch.zeros_like(pred)
        for n in range(nnuc+1):
            if input.grad is not None:
                #sets componnents to zero if not already zero
                input.grad.data.zero_()

            pred[:, n].backward(torch.ones_like(pred[:,n]), retain_graph=True)
            dpndx[:, n] = input.grad.clone()[:, 0]
            input.grad.data.zero_()

        #L = nn.MSELoss() # this produces infinte loss because the values start off so different.
        L = nn.L1Loss()
        loss = L(dpndx, target[:, nnuc+1:])
        # print('dpndx')
        # print(dpndx)
        # print('target')
        # print(target[:, nnuc+1:])
        return loss

    def loss_mass_fraction(pred):
        return torch.abs(1 - torch.sum(pred[:, :nnuc]))


    def loss_pure(pred, actual, log_option = False):
        L = nn.MSELoss()

        #implement log option to scale loss logarithmly for low values

        return L(pred, actual[:, :nnuc+1])





    def objective(trial):

        # Generate the model.
        model = define_model(trial).to(DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZERS)
        lr = trial.suggest_float("lr", LEARNING_RATE[0], LEARNING_RATE[1], log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)



        def criterion(data, pred, actual):
            #I still don't understand this one but i think don does

            loss1 = rms_weighted_error(pred, actual[:, :nnuc+1], actual[:, :nnuc+1])
            #physics informed loss
            loss2 = loss_pinn(data, pred, actual)
            #sum of mass fractions must be 1
            loss3 = loss_mass_fraction(pred)
            #difference in state variables vs prediction.
            loss4 = loss_pure(pred, actual, log_option = False)

            return loss1 + loss2 + loss3 + loss4


        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break
                data.requires_grad=True

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(data, output, target)
                loss.backward()
                optimizer.step()


            #Validation of model
            model.eval()
            #we save the largest MSE of the set as the accuracy of the trial.
            accuracy = -np.inf
            L = nn.MSELoss()
            with torch.no_grad():
                 for batch_idx, (data, target) in enumerate(test_loader):
                        #data.requires_grad=True

                        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                        output = model(data)


                        local_accuracy = criterion(data, output, target)

                        if local_accuracy > accuracy:
                            accuracy = local_accuracy

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study.best_params



def print_h_opt_results(dict):
    print("~~~Hyperparamter Optimization Results:~~~")
    print(f"Optimizer Chosen: {dict['optimizer']} with learning rate: {dict['lr']}")
    print(f"Neural Network with {dict['n_layers']} layers")
    for i in range(dict['n_layers']):
        print(f"Layer {i+1}: Number of units: {dict['n_units_l{}'.format(i)]} \
         Dropout Rate: {dict['dropout_l{}'.format(i)]}")


    # n_units.append(hyper_results["n_units_l{}".format(i)])
    # p_dropouts.append(hyper_results["dropout_l{}".format(i)])
    #
    #
    # n_layers = dict['n_layers']
