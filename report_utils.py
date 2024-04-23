import numpy as np
import  numpy.fft as fft
import xarray as xr
from model_me5311 import DataProcessor

# dimensions of data
B = 16071
H = 101 
W = 161
split_idx = 15342 # 2021-01-01

t_scale = 4 / (365*4 + 1)
yr_start = 1979

# load data
ds_slp = xr.open_dataset('data/slp.nc')["msl"] 
ds_t2m = xr.open_dataset('data/t2m.nc')["t2m"]
ds_slp = np.array(ds_slp)
ds_t2m = np.array(ds_t2m)

# split into train and test set
slp_train, slp_test = ds_slp[:split_idx], ds_slp[split_idx:]
t2m_train, t2m_test = ds_t2m[:split_idx], ds_t2m[split_idx:]
del ds_slp, ds_t2m

# rescale the date such that each unit represents a whole year
time_arr = np.arange(B) * t_scale
t_train, t_test = time_arr[:split_idx], time_arr[split_idx:]


train_data = slp_train.reshape(-1, H*W)
test_data = slp_test.reshape(-1, H*W)
threshold = 1.5e7
unit = "Pa"

# train_data = t2m_train.reshape(-1, H*W)
# test_data = t2m_test.reshape(-1, H*W)
# threshold = 6e4
# unit = "℃"