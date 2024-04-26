import numpy as np
import  numpy.fft as fft
import xarray as xr
from model_me5311 import DataProcessor
import matplotlib.pyplot as plt

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

model = DataProcessor()

n_components = 10
modes_filtered = model.fit(t_train, train_data, 
                           n_components=n_components, 
                           threshold=threshold, 
                           normalize=True
                           )

modes_train_pred = model.pred_modes(t_train)

modes_test_pred = model.pred_modes(t_test)
modes_train_pred = model.pred_modes(t_train)

from matplotlib.ticker import MaxNLocator

baseline_est = np.mean(train_data, axis=0)
err_baseline = np.abs(baseline_est - test_data)
err_baseline_mean = err_baseline.mean(axis=1)
err_baseline_median = np.median(err_baseline, axis=1)
err_baseline_max = err_baseline.max(axis=1)

test_pred = model.inversePCA(t_test, modes_test_pred).reshape(-1, H, W)
err = np.abs(test_pred - test_data.reshape(-1, H, W))
err_mean = np.mean(err, axis=(1,2))
err_median = np.median(err, axis=(1,2))
err_max = np.max(err, axis=(1, 2))
print("Mean error: ", err.mean())
print("Max error: ", err.max())
w, b = np.polyfit(t_test, err_mean, 1)
print("growth rate: ", w)


test_pca = model.pca_model.transform(test_data.reshape(-1, H*W))
test_denoised = model.pca_model.inverse_transform(test_pca) # denoise the test data with pca
err_denoising = np.abs(test_denoised - test_data)
err_denoising_mean = np.mean(err_denoising, axis=1)
err_denoising_median = np.median(err_denoising, axis=(1,))
err_denoising_max = np.max(err_denoising, axis=(1, ))


fig, axs = plt.subplots(2, 1)
axs[0].plot(t_test+yr_start, err_mean)
axs[0].plot(t_test+yr_start, err_baseline_mean)
# plt.plot(t_test+yr_start, err_denoising_mean)
axs[0].plot(t_test+yr_start, w * t_test + b, c='r')
# plt.plot(t_test+yr_start, np.mean(np.abs(test_pred - baseline_est.reshape(-1, H, W)),axis=(1,2)))
# axs[0].set_xlabel("Year")
axs[0].set_ylabel(f"Error ({unit})")
axs[0].legend(["model", "baseline", "trend", "denoising"])
axs[0].set_title("Mean Absolute Error")

train_data = t2m_train.reshape(-1, H*W)
test_data = t2m_test.reshape(-1, H*W)
threshold = 6e4
unit = "℃"

n_components = 10
modes_filtered = model.fit(t_train, train_data, 
                           n_components=n_components, 
                           threshold=threshold, 
                           normalize=True
                           )

modes_train_pred = model.pred_modes(t_train)
modes_test_pred = model.pred_modes(t_test)
modes_train_pred = model.pred_modes(t_train)

from matplotlib.ticker import MaxNLocator

baseline_est = np.mean(train_data, axis=0)
err_baseline = np.abs(baseline_est - test_data)
err_baseline_mean = err_baseline.mean(axis=1)
err_baseline_median = np.median(err_baseline, axis=1)
err_baseline_max = err_baseline.max(axis=1)

test_pred = model.inversePCA(t_test, modes_test_pred).reshape(-1, H, W)
err = np.abs(test_pred - test_data.reshape(-1, H, W))
err_mean = np.mean(err, axis=(1,2))
err_median = np.median(err, axis=(1,2))
err_max = np.max(err, axis=(1, 2))
# print("Mean error: ", err.mean())
print("Max error: ", err.max())
w, b = np.polyfit(t_test, err_mean, 1)
print("growth rate: ", w)


# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].plot(t_test+yr_start, err_mean)
axs[1].plot(t_test+yr_start, err_baseline_mean)
# plt.plot(t_test+yr_start, err_denoising_mean)
axs[1].plot(t_test+yr_start, w * t_test + b, c='r')
# plt.plot(t_test+yr_start, np.mean(np.abs(test_pred - baseline_est.reshape(-1, H, W)),axis=(1,2)))
axs[1].set_xlabel("Year")
axs[1].set_ylabel(f"Error ({unit})")
axs[1].legend(["model", "baseline", "trend", "denoising"])
# axs[1].set_title("Mean Absolute Error")
plt.show()
