# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# トーラスのパラメータ
R = 1  # トーラスの中心から軸までの距離
r = 0.3  # トーラスの断面円の半径
theta_res = 100  # 断面円の分割数
phi_res = 100  # 軸周りの分割数

# パラメータ値の生成
theta = np.linspace(0, 2 * np.pi, theta_res)
phi = np.linspace(0, 2 * np.pi, phi_res)
theta, phi = np.meshgrid(theta, phi)
c, a = R + r * np.cos(theta), r * np.sin(theta)

# トーラスの座標計算
x = c * np.cos(phi)
y = c * np.sin(phi)
z = a

# 三次元プロット
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(x, y, z, color="blue", alpha=0.3)

# グラフ表示
plt.show()

#%%
now = datetime.datetime.now()
filename = 'result_SOM_image_torus' + now.strftime('%Y%m%d_%H%M%S') + '.csv'


# 座標をデータフレームに格納
df = pd.DataFrame(np.array([x.flatten(), y.flatten(), z.flatten()]).T, columns=["x", "y", "z"])
df.to_csv("torus.csv", index=False)
# %%
