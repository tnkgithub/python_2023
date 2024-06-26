# %%
import geodesic_dome as gd
import matplotlib.pyplot as plt
import pandas as pd

# %%
vertices, triangels, adj_list = gd.create_geodesic_dome(3)

# %%
print(vertices.shape)

# %%
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="None", subplot_kw={"projection": "3d"}
)
plt.gca().axis("off")
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]
# ax.plot_wireframe(x, y, z, alpha=0.5)  # くり抜き曲面
ax.scatter(x, y, z, alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.suptitle("spherical surface", fontsize=10)
ax.set_aspect("equal")
plt.show()

# %%
# 座標をデータフレームに格納
df = pd.DataFrame(vertices, columns=["x", "y", "z"])
df.to_csv("geodesic_dome.csv", index=False)
