from report_utils import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np



# Bar plot for mean error comparison
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

categories = ['PCA err\n(SLP)', 'Baseline err\n(SLP)', 'SSM err\n(SLP)', 'PCA err\n(T2M)', 'Baseline err\n(T2M)', 'SSM err\n(T2M)']
x = np.arange(len(categories))  # positions for all categories
bar_width = 0.78


# Adjust bar positions for grouping
slp_positions = x[:3]  # SLP positions
t2m_positions = x[3:]  # T2M positions

# Bar colors
colors = ['#ff7f0e', '#1f77b4', '#2ca02c']

slp = [206.63, 131.3446, 101.4946]
t2m = [1.11, 0.6632, 0.5128]
bars_slp_baseline = ax1.bar(slp_positions[0] + 0.3*bar_width, slp[0], bar_width, label='SLP Baseline', color=colors[0],  capsize=10)
bars_slp_fft = ax1.bar(slp_positions[1], slp[1], bar_width, label='SLP FFT', color=colors[1],  capsize=10)
bars_slp_predict = ax1.bar(slp_positions[2] - 0.3*bar_width, slp[2], bar_width, label='SLP PREDICT', color=colors[2], capsize=10)

# T2M bars
bars_t2m_baseline = ax2.bar(t2m_positions[0] + 0.3*bar_width, t2m[0], bar_width, label='T2M Baseline', color=colors[0],  capsize=10)
bars_t2m_fft = ax2.bar(t2m_positions[1], t2m[1], bar_width, label='T2M FFT', color=colors[1],  capsize=10)
bars_t2m_predict = ax2.bar(t2m_positions[2] - 0.3*bar_width, t2m[2], bar_width, label='T2M PREDICT', color=colors[2], capsize=10)
# ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Error (Pa)', color='blue')
ax2.set_ylabel('Mean Error (℃)', color='red')

ax1.set_xticks(x)
ax1.set_xticklabels(categories)

# Customizing legend placement
# ax1.legend(loc='upper left', bbox_to_anchor=(0.3,1))
# ax2.legend(loc='upper left', bbox_to_anchor=(0.8,1))

for bar in bars_slp_baseline:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

for bar in bars_slp_fft:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

for bar in bars_slp_predict:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

for bar in bars_t2m_baseline:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

for bar in bars_t2m_fft:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

for bar in bars_t2m_predict:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')

plt.title('Prediction accuracy of the SSM model')
plt.show()
