'''
スライド用
'''
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# 半径を指定
r = 1.0

# ラジアンを作成
# q: 0.0 ~ 2.0*np.piとは何か
# a: 0.0 ~ 2.0*np.piの範囲で81個の等間隔の数値を生成
t = np.linspace(start=0.0, stop=2.0 * np.pi, num=81)
u = np.linspace(start=0.0, stop=2.0 * np.pi, num=81)

# 格子点を作成
T, U = np.meshgrid(t, u)

# 球面座標を直交座標に変換
X = r * np.sin(T) * np.cos(U)
Y = r * np.sin(T) * np.sin(U)
Z = r * np.cos(T)

# %%
XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

# %%
# 球面を作図
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="white", subplot_kw={"projection": "3d"}
)
ax.plot_wireframe(X, Y, Z, alpha=0.5)  # くり抜き曲面
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.suptitle("spherical surface", fontsize=20)
ax.set_aspect("equal")
plt.show()

#%%
# 座標をデータフレームに格納
df = pd.DataFrame(XYZ, columns=["x", "y", "z"])
df.to_csv("geocentric_cartesian_coordinates.csv", index=False)
# %%
