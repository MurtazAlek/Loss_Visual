from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
surf_name = "test_loss"
result_file_path = "../result/3d_surface_file_ResNet20_SHCUT.h5"
#
# with h5py.File("../3d_surface_file_ResNet20_SHCUT.h5",'r') as f:
#     Z_LIMIT = 10
#
#     x = np.array(f['xcoordinates'][:])
#     y = np.array(f['ycoordinates'][:])
#
#     X, Y = np.meshgrid(x, y)
#     Z = np.array(f[surf_name][:])
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("f(x, y)")
#     #ax.plot_wireframe(X, Y, Z)
#     ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
#     plt.show()
#
#
#
# # with h5py.File(surface_path, "r") as f:
# #     print("Keys: %s" % f.keys())
# #     a_group_key = list(f.keys())[2]
# #     data = list(f[a_group_key])
# #     print(data)
#
