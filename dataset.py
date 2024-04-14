import numpy as np
import torch
from torch.utils.data import Dataset
import os


SLP_ERR_SCALE = 300
T2M_ERR_SCALE = 1.5

class ME5311Dataset(Dataset):
    def __init__(self, dir="data", train=True, type="slp", t_size=11, use_err=True):
        train_str = "_train_" if train else "_test_"
        err_str = "err" if use_err else ""
        file_name = type + train_str + err_str + ".npy"
        file_path = os.path.join(dir, file_name)

        if type == "slp" and use_err:
            scale = SLP_ERR_SCALE
        elif type == "t2m" and use_err:
            scale = T2M_ERR_SCALE
        
        # Load the numpy file
        self.data = np.load(file_path) / SLP_ERR_SCALE
        self.data = np.clip(self.data, -1, 1)
        self.t_size = t_size 
         
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data) - self.t_size + 1
    
    def __getitem__(self, idx):
        # Return the sample at the given index
        sample = self.data[idx:idx + self.t_size]
        return torch.tensor(sample[:-1], dtype=torch.float32), torch.tensor(sample[1:], dtype=torch.float32)
    
if __name__ == "__main__":
    ME5311Dataset()


