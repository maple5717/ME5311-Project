import numpy as np
import  numpy.fft as fft
import xarray as xr
from model_me5311 import DataProcessor
from report_utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

n_components = 10
model = DataProcessor()

modes_filtered = model.fit(t_train, train_data, 
                           n_components=n_components, 
                           threshold=threshold, 
                           normalize=True
                           )

modes_train_pred = model.pred_modes(t_train)
modes_test_pred = model.pred_modes(t_test)

test_pca = model.pca_model.transform(test_data.reshape(-1, H*W))
test_denoised = model.pca_model.inverse_transform(test_pca) # denoise the test data with pca
err_denoising = np.abs(test_denoised - test_data)
err_denoising_mean = np.mean(err_denoising, axis=1)
err_denoising_median = np.median(err_denoising, axis=(1,))
err_denoising_max = np.max(err_denoising, axis=(1, ))


baseline_est = np.mean(train_data, axis=0)
err_baseline = np.abs(baseline_est - test_data)
err_baseline_mean = err_baseline.mean(axis=1)
err_baseline_median = np.median(err_baseline, axis=1)
err_baseline_max = err_baseline.max(axis=1)

test_pred = model.inversePCA(t_test, modes_test_pred).reshape(-1, H, W)
err = np.abs(test_pred - test_data.reshape(-1, H, W))
err_mean = np.mean(err, axis=(1,2)) # daily error
print("Mean error: ", err.mean())
print("25% quantile of error per day: ",np.quantile(err_mean, 0.25))
print("75% quantile of error per day: ",np.quantile(err_mean, 0.75))
w, b = np.polyfit(t_test, err_mean, 1)
print("growth rate: ", w)

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(t_test+yr_start, err_mean)
plt.plot(t_test+yr_start, err_baseline_mean)
plt.plot(t_test+yr_start, err_denoising_mean)
plt.plot(t_test+yr_start, w * t_test + b, c='r')
# plt.plot(t_test+yr_start, np.mean(np.abs(test_pred - baseline_est.reshape(-1, H, W)),axis=(1,2)))
plt.xlabel("Year")
plt.ylabel(f"Error ({unit})")
plt.legend(["model", "baseline", "trend", "denoising"])
plt.title("Mean error")



