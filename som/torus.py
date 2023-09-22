# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# トーラスのパラメータ
R = 1  # トーラスの中心から軸までの距離
r = 0.3  # トーラスの断面円の半径
theta_res = 34  # 断面円の分割数
phi_res = 59  # 軸周りの分割数

# パラメータ値の生成
theta = np.linspace(0, 2 * np.pi, theta_res)
phi = np.linspace(0, 2 * np.pi, phi_res)
theta, phi = np.meshgrid(theta, phi)
c, a = R + r * np.cos(theta), r * np.sin(theta)

# トーラスの座標計算
x = c * np.cos(phi)
y = c * np.sin(phi)
z = a

#%%
# 三次元プロット
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="None", subplot_kw={"projection": "3d"}
)
plt.gca().axis("off")
ax.scatter(x, y, z, alpha=0.3)
ax.set_aspect("equal")

# グラフ表示
plt.show()

# %%


# 座標をデータフレームに格納
df = pd.DataFrame(
    np.array([x.flatten(), y.flatten(), z.flatten()]).T, columns=["x", "y", "z"]
)
df.to_csv("./result_vertices/torus_43_46.csv", index=False)


# %%
# ２次元平面を作成（スライド用）

x = np.arange(40)
y = np.arange(40)
X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(15, 15), facecolor="None")
plt.gca().axis("off")
ax.scatter(X, Y, alpha=1, s=50)
ax.set_aspect("equal")
plt.show()

# %%
# 筒形のパラメータ（スライド用）
R = 1  # 筒形の中心から軸までの距離
r = 0.2  # 筒形の断面円の半径
height = 1  # 筒形の高さ
theta_res = 50  # 断面円の分割数
z_res = 50  # 高さ方向の分割数を1に設定

# パラメータ値の生成
theta = np.linspace(0, 2 * np.pi, theta_res)
z = np.linspace(-height / 2, height / 2, z_res)
theta, z = np.meshgrid(theta, z)

# 筒形の座標計算
x = (R + r * np.cos(theta)) * np.cos(z)
y = (R + r * np.cos(theta)) * np.sin(z)
z = r * np.sin(theta)

# 三次元プロット
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="None", subplot_kw={"projection": "3d"}
)
plt.gca().axis("off")
ax.scatter(x, y, z, alpha=0.5)
ax.set_aspect("equal")
# グラフ表示
plt.show()

# %%
