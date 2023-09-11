import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F

class DataGenerator():
    def __init__(self, ds, var_dict, lead_time, batch_size, shuffle=True,  device=None):

        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = []
        data1 = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            if var != 'tp':
                try:
                    data.append(ds[var].sel(level=levels))
                except ValueError:
                    data.append(ds[var].expand_dims({'level': generic_level}, 1))
                except KeyError:
                    data.append(ds[var])
            else :
                    data1.append(ds[var])
    
        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.data1 = xr.concat(data1, dim='time').transpose('time', 'lat', 'lon')
        self.n_samples1 = self.data1.isel(time=slice(0, -lead_time)).shape[0]
        self.n_samples1 = self.n_samples1 - self.n_samples1 % self.batch_size # adjust n_samples to be a multiple of batch_size
        self.init_time1 = self.data1.isel(time=slice(None, -lead_time)).time
        self.valid_time1 = self.data1.isel(time=slice(lead_time, None)).time
        self.on_epoch_end1()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples1 / self.batch_size))
    
    def to_tensor(self, x):
        # Convert a numpy array to a PyTorch tensor
        return torch.from_numpy(x).float().to(self.device)
    
    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data1.isel(time=idxs + self.lead_time).values
        y=y*1000
        X = self.to_tensor(X)
        y = self.to_tensor(y)
        y = F.relu(y)

        return X, y

    def on_epoch_end1(self):
        self.idxs = np.arange(self.n_samples1)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        
        remainder = self.n_samples1 % self.batch_size
        if remainder != 0:
            self.idxs = self.idxs[:self.n_samples1 - remainder]
    
