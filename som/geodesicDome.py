# %%
import geodesic_dome as gd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# %%
vertices, triangels, adj_list = gd.create_geodesic_dome(4)

# %%
print(vertices.shape)

# %%
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="white", subplot_kw={"projection": "3d"}
)

x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]
ax.plot_wireframe(x, y, z, alpha=0.5)  # くり抜き曲面
# ax.scatter(x, y, z, alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.suptitle("spherical surface", fontsize=10)
ax.set_aspect("equal")
plt.show()

# %%
