from report_utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

n_components = 10
model = DataProcessor()

# Model fitting for SLP
modes_filtered = model.fit(t_train, train_data,
                           n_components=n_components,
                           threshold=threshold,
                           normalize=True)

# Prediction on training and testing data for SLP
modes_train_pred = model.pred_modes(t_train)
modes_test_pred = model.pred_modes(t_test)

# PCA transformation for test data and denoising for SLP
test_pca = model.pca_model.transform(test_data.reshape(-1, H*W))
test_denoised = model.pca_model.inverse_transform(test_pca)
err_denoising = np.abs(test_denoised - test_data)
err_denoising_mean = np.mean(err_denoising, axis=1)

# Baseline estimation and error calculation for SLP
baseline_est = np.mean(train_data, axis=0)
err_baseline = np.abs(baseline_est - test_data)
err_baseline_mean = np.mean(err_baseline, axis=1)

# Prediction from FFT model for SLP
test_pred = model.inversePCA(t_test, modes_test_pred).reshape(-1, H, W)
err = np.abs(test_pred - test_data.reshape(-1, H, W))
err_mean = np.mean(err, axis=(1,2))

# Repeat for T2M
modes_filtered_t2m = model.fit(t_train, train_data_t2m,
                               n_components=n_components,
                               threshold=threshold_t2m,
                               normalize=True)

modes_train_pred_t2m = model.pred_modes(t_train)
modes_test_pred_t2m = model.pred_modes(t_test)

test_pca_t2m = model.pca_model.transform(test_data_t2m.reshape(-1, H*W))
test_denoised_t2m = model.pca_model.inverse_transform(test_pca_t2m)
err_denoising_t2m = np.abs(test_denoised_t2m - test_data_t2m)
err_denoising_mean_t2m = np.mean(err_denoising_t2m, axis=1)

baseline_est_t2m = np.mean(train_data_t2m, axis=0)
err_baseline_t2m = np.abs(baseline_est_t2m - test_data_t2m)
err_baseline_mean_t2m = np.mean(err_baseline_t2m, axis=1)

test_pred_t2m = model.inversePCA(t_test, modes_test_pred_t2m).reshape(-1, H, W)
err_t2m = np.abs(test_pred_t2m - test_data_t2m.reshape(-1, H, W))
err_mean_t2m = np.mean(err_t2m, axis=(1,2))

# Bar plot for mean error comparison
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

categories = ['Baseline (SLP)', 'PCA model (SLP)', 'Baseline (T2M)', 'PCA model (T2M)']
x = np.arange(len(categories))  # positions for all categories
bar_width = 0.6

# Error values
mean_errors_slp = [err_baseline_mean.mean(), err_mean.mean()]
mean_errors_t2m = [err_baseline_mean_t2m.mean(), err_mean_t2m.mean()]

# Error bars configuration
error_bars_slp = [[mean_errors_slp[i] - np.quantile(err_baseline_mean if i == 0 else err_mean, 0.25),
                   np.quantile(err_baseline_mean if i == 0 else err_mean, 0.75) - mean_errors_slp[i]] for i in range(2)]
error_bars_t2m = [[mean_errors_t2m[i] - np.quantile(err_baseline_mean_t2m if i == 0 else err_mean_t2m, 0.25),
                   np.quantile(err_baseline_mean_t2m if i == 0 else err_mean_t2m, 0.75) - mean_errors_t2m[i]] for i in range(2)]

# Adjust bar positions for grouping
slp_positions = x[:2]  # SLP positions
t2m_positions = x[2:]  # T2M positions

# Bar colors
colors = ['gray', 'orange']

# SLP bars
bars_slp_baseline = ax1.bar(slp_positions[0] + 0.325*bar_width, mean_errors_slp[0], bar_width, label='SLP Baseline', color=colors[0], yerr=np.array([error_bars_slp[0]]).T, capsize=10)
bars_slp_fft = ax1.bar(slp_positions[1] - 0.325*bar_width, mean_errors_slp[1], bar_width, label='SLP FFT', color=colors[1], yerr=np.array([error_bars_slp[1]]).T, capsize=10)

# T2M bars
bars_t2m_baseline = ax2.bar(t2m_positions[0] + 0.325*bar_width, mean_errors_t2m[0], bar_width, label='T2M Baseline', color=colors[0], yerr=np.array([error_bars_t2m[0]]).T, capsize=10)
bars_t2m_fft = ax2.bar(t2m_positions[1] - 0.325*bar_width, mean_errors_t2m[1], bar_width, label='T2M FFT', color=colors[1], yerr=np.array([error_bars_t2m[1]]).T, capsize=10)

ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Absolute Error (Pa)', color='blue')
ax2.set_ylabel('Mean Absolute Error (℃)', color='red')

ax1.set_xticks(x)
ax1.set_xticklabels(categories)

# Customizing legend placement
# ax1.legend(loc='upper left', bbox_to_anchor=(0.3,1))
# ax2.legend(loc='upper left', bbox_to_anchor=(0.8,1))

for bar in bars_slp_baseline:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2*1.25, yval, round(yval, 2), va='bottom')

for bar in bars_slp_fft:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2*1.25, yval, round(yval, 2), va='bottom')

for bar in bars_t2m_baseline:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2*1.25, yval, round(yval, 2), va='bottom')

for bar in bars_t2m_fft:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2*1.25, yval, round(yval, 2), va='bottom')

plt.title('Performance of the PCA model on the test dataset')
plt.show()
