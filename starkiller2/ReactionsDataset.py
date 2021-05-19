import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
#import torchvision.transforms.functional as TF

# Standardize a data array
def Standardize(x, mean=None, std=None, deriv=False): 
    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    
    if deriv:
        y = x / std
    else:
        y = (x - mean) / std
        
    return y

# Normalize a data array
def Normalize(x, x_min=None, x_max=None, deriv=False): 
    if x_min is None:
        x_min = np.min(x, axis=0)
    if x_max is None:
        x_max = np.max(x, axis=0)
        
    if deriv:
        y = x / (x_max - x_min)
    else:
        y = (x - x_min) / (x_max - x_min)
        
    return y

# Convert numpy data to pytorch dataset
class ReactionsDataset(Dataset): 
    """Reactions dataset."""

    def __init__(self, x, y, dydt, system, normalize=False):
        """
        Args:
            x (array): solutions at t0. (input)
            y (array): solutions after dt. (output)
            dydt (array): derivative of solutions wrt time.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.net_itemp = system.network.net_itemp
        self.net_ienuc = system.network.net_ienuc
        
        # copy variables
        self.x = x.copy()
        self.y = y.copy()
        self.dydt = dydt.copy()
        
        # normalization 
        self.dt_scale = max(x[:,0])
        self.x[:,0] = self.x[:,0] / self.dt_scale
        self.dydt *= self.dt_scale
        
        if normalize:
            # normalize all input variables
            x_min = np.min(self.x[:,1:], axis=0)
            x_max = np.max(self.x[:,1:], axis=0)
            self.x[:,1:] = Normalize(self.x[:,1:], x_min=x_min, x_max=x_max)
            self.y = Normalize(self.y, x_min=x_min, x_max=x_max)
            self.dydt = Normalize(self.dydt, x_min=x_min, x_max=x_max, deriv=True)
        else:
            # standardize temperature and energy
            self.temp_mean = np.mean(x[:,self.net_itemp+1], axis=0)
            self.temp_std = np.std(x[:,self.net_itemp+1], axis=0)
            self.enuc_mean = np.mean(x[:,self.net_ienuc+1], axis=0)
            self.enuc_std = np.std(x[:,self.net_ienuc+1], axis=0)
        
            # normalize temperature and energy
            self.x[:,self.net_itemp+1] = Standardize(x[:,self.net_itemp+1], 
                                                     mean=self.temp_mean, std=self.temp_std)
            self.x[:,self.net_ienuc+1] = Standardize(x[:,self.net_ienuc+1], 
                                                     mean=self.enuc_mean, std=self.enuc_std)
            self.y[:,self.net_itemp] = Standardize(y[:,self.net_itemp], 
                                                   mean=self.temp_mean, std=self.temp_std)
            self.y[:,self.net_ienuc] = Standardize(y[:,self.net_ienuc], 
                                                   mean=self.enuc_mean, std=self.enuc_std)
        
            self.dydt[:,self.net_itemp] = Standardize(self.dydt[:,self.net_itemp], 
                                                      mean=self.temp_mean, std=self.temp_std, 
                                                      deriv=True)
            self.dydt[:,self.net_ienuc] = Standardize(self.dydt[:,self.net_ienuc], 
                                                      mean=self.enuc_mean, std=self.enuc_std, 
                                                      deriv=True)
        
        # convert to tensors
        # we also want to propagate gradients through y, dydt, and x
        self.x = torch.tensor(self.x, requires_grad=True, dtype=torch.float)
        self.y = torch.tensor(self.y, requires_grad=True, dtype=torch.float)
        self.dydt = torch.tensor(self.dydt, requires_grad=True, 
                                 dtype=torch.float)  # used in computing loss1 later

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # get items at indices idx
        x = self.x[idx,:]
        y = self.y[idx,:]
        dydt = self.dydt[idx,:]
        
        sample = {'x': x, 'y': y, 'dydt': dydt, 'idx': idx}

        return sample
    
    def _getparam_(self):
        # get normalization parameters
        param = {'T_mean': self.temp_mean, 'T_std': self.temp_std, 
                 'E_mean': self.enuc_mean, 'E_std': self.enuc_std}
        
        return param