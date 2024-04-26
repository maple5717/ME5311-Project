import numpy as np
import matplotlib.pyplot as plt

def git_figure(path):
    data = np.load(path)
    pred = data['pred']
    truth = data['truth']

    return pred, truth


images = [None] * 4
t=20

slp_pred, slp_truth = git_figure("ssm_slp.npz")
print(slp_pred.shape, slp_truth.shape)
k = 25
images[0] = [slp_pred[k], slp_truth[k+t],  slp_truth[k+t-1]] # prev, truch, pred
print(slp_pred[k].shape, slp_truth[k+t].shape)
k = 30
images[1] = [slp_pred[k], slp_truth[k+t],  slp_truth[k+t-1] ]# prev, truch, pred

slp_pred, slp_truth = git_figure("ssm_t2m.npz")

k = 20
images[2] = [slp_pred[k], slp_truth[k+t],  slp_truth[k+t-1]] # prev, truch, pred
k = 30
images[3] = [slp_pred[k], slp_truth[k+t],  slp_truth[k+t-1]] # prev, truch, pred

fig, axs = plt.subplots(3, 4, figsize=(12, 10))

# Plot the figures in the 1st and 3rd rows
# for i, ax in enumerate(axs.flat):
#     if i // 3 % 2 == 0:  # Check if it's the 1st or 3rd row
#         ax.imshow()
#         title = f'{i//3 + 1}{"abc"[i%2]}'
#         ax.set_title(title)
for i in range(4):
    scale = 300 if i <= 1 else 1.5
    name = "SLP" if i <= 1 else "T2M"
    titles = ["Model Prection", "Ground Truth", "Previous Timestep"]
    for j in range(3):
        im=axs[i][j].imshow(images[i][j], cmap="RdBu", vmin=-scale, vmax=scale)
        axs[i][j].set_xticks([])
        axs[i][j].set_yticks([])

        if i == 0 or i== 2:
            axs[i][j].set_title(f"{titles[j]} ({name})")

        if j == 2:
            cbar = fig.colorbar(im, ax=axs[i][j])

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
