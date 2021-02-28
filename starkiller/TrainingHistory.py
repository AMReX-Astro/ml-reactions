import numpy as np
import h5py

class TrainingHistory(object):
    def __init__(self):
        # system data
        self.nspec = None
        self.net_itemp = None
        self.net_ienuc = None

        # inputs & truth for training & test data
        self.training_inputs = []
        self.training_truth = []
        self.training_truth_rhs = []
        self.test_inputs = []
        self.test_truth = []

        # arrays for accumulating the epoch index and losses during training
        self.epochs = []
        self.losses = []
        self.losses0 = []
        self.losses1 = []
        self.test_losses = []

        # arrays for saving prediction & prediction RHS during training
        self.model_history = []
        self.model_rhs_history = []
        self.model_grad_history = []
        self.test_model_history = []

    def set_training_test_data(self, system,
                               training_inputs, training_truth,
                               training_truth_rhs,
                               test_inputs, test_truth):
        # inputs & truth for training & test data
        self.training_inputs = training_inputs
        self.training_truth = training_truth
        self.training_truth_rhs = training_truth_rhs
        self.test_inputs = test_inputs
        self.test_truth = test_truth

        # system data
        self.nspec = system.network.nspec
        self.net_itemp = system.network.net_itemp
        self.net_ienuc = system.network.net_ienuc

    def save_history(self, history_file="training_history.h5"):
        # create HDF5 file
        history = h5py.File(history_file, "w")

        # save system data
        history["nspec"] = np.array([self.nspec])
        history["net_itemp"] = np.array([self.net_itemp])
        history["net_ienuc"] = np.array([self.net_ienuc])

        # save epoch numbers and losses
        history["epochs"] = np.array(self.epochs)
        history["model_losses"] = np.array(self.losses)
        history["prediction_losses"] = np.array(self.losses0)
        history["rhs_losses"] = np.array(self.losses1)
        history["test_losses"] = np.array(self.test_losses)

        # save inputs, truth for training data set
        history["training_inputs"] = np.array(self.training_inputs)
        history["training_truth"] = np.array(self.training_truth)
        history["training_truth_rhs"] = np.array(self.training_truth_rhs)

        # save inputs, truth for test data set
        history["test_inputs"] = np.array(self.test_inputs)
        history["test_truth"] = np.array(self.test_truth)

        # save training & test model predictions & rhs
        history["training_model_history"] = np.array(self.model_history)
        history["training_model_rhs_history"] = np.array(self.model_rhs_history)
        history["training_model_grad_history"] = np.array(self.model_grad_history)
        history["test_model_history"] = np.array(self.test_model_history)

        # close HDF5 file
        history.close()

    def load_history(self, history_file="training_history.h5"):
        # open HDF5 file
        history = h5py.File(history_file, "r")

        # load system data
        self.nspec = np.array(history["nspec"])[0]
        self.net_itemp = np.array(history["net_itemp"])[0]
        self.net_ienuc = np.array(history["net_ienuc"])[0]

        # load epoch numbers and losses
        self.epochs = np.array(history["epochs"])
        self.losses = np.array(history["model_losses"])
        self.losses0 = np.array(history["prediction_losses"])
        self.losses1 = np.array(history["rhs_losses"])
        self.test_losses = np.array(history["test_losses"])

        # load inputs, truth for training data set
        self.training_inputs = np.array(history["training_inputs"])
        self.training_truth = np.array(history["training_truth"])
        self.training_truth_rhs = np.array(history["training_truth_rhs"])

        # load inputs, truth for test data set
        self.test_inputs = np.array(history["test_inputs"])
        self.test_truth = np.array(history["test_truth"])

        # load training & test model predictions & rhs
        self.model_history = np.array(history["training_model_history"])
        self.model_rhs_history = np.array(history["training_model_rhs_history"])
        self.model_grad_history = np.array(history["training_model_grad_history"])
        self.test_model_history = np.array(history["test_model_history"])

        # close HDF5 file
        history.close()