import optuna
import torch
import torch.nn as nn
import os
import torch.optim as optim
import numpy as np

def do_h_opt(train_loader, test_loader, BATCHSIZE, CLASSES, EPOCHS,
                      LOG_INTERVAL, N_TRAIN_EXAMPLES, N_VALID_EXAMPLES,
                      LAYERS, UNITS, DROPOUT_RATE, LEARNING_RATE, OPTIMIZERS,
                      n_trials, timeout):

    DEVICE = torch.device("cpu")
    DIR = os.getcwd()


    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", LAYERS[0], LAYERS[1])

        layers = []

        in_features = 16
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), UNITS[0], UNITS[1])
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), DROPOUT_RATE[0], DROPOUT_RATE[1])
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, CLASSES))
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
