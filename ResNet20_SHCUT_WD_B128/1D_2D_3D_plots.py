from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
import pandas as pd


dataset = pd.read_csv("ResNet20_SHCUT_model_history_log.csv", delimiter=",")


fig_0 = plt.figure()
plt.plot(dataset['epoch'],dataset['loss'], label='train_loss')
plt.plot(dataset['epoch'],dataset['val_loss'], label='val_loss')
plt.legend()
fig_0.savefig(fname='loss_and_val_loss_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')

fig_00 = plt.figure()
plt.plot(dataset['epoch'],dataset['Acc'], label='train_Acc')
plt.plot(dataset['epoch'],dataset['val_Acc'], label='val_Acc')
plt.legend()
fig_0.savefig(fname='Acc_and_val_Acc_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')




surf_name = "test_loss"

with h5py.File(r'D:\Loss_Visual\ResNet20_SHCUT_WD_B128\3d_surface_file_ResNet20_SHCUT.h5','r') as f:

    Z_LIMIT = 10

    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])

    X, Y = np.meshgrid(x, y)
    Z = np.array(f[surf_name][:])
    # Z[Z > Z_LIMIT] = Z_LIMIT
    # Z = np.log(Z)  # logscale


    fig_1 = plt.figure()
    ax = Axes3D(fig_1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    fig_1.savefig(fname='3D_surface_plot_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')



    fig_2 = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer')
    plt.clabel(CS, inline=1, fontsize=8, colors='red')
    fig_2.savefig(fname='2D_contor_plot_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')


    fig_3 = plt.figure()
    CS = plt.contourf(X, Y, Z, cmap='summer')
    plt.clabel(CS, inline=1, fontsize=8, colors='red')
    fig_3.savefig(fname='2D_contorf_plot_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')


    plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=0.55, vmax=0.85,
                               xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(fname='HeatMap_ResNet20_SHCUT.pdf', dpi=300, bbox_inches='tight', format='pdf')


plt.show()


