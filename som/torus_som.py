# %%
import numpy as np
import pandas as pd
import os


from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

# %%
# データの読み込み
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values

# データの正規化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = scaler.fit_transform(df)

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# %%
XYZ = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

# %%
print(XYZ.shape)

# %%
# 初期値の設定
n_rows = 200  # 縦方向のノード数
n_cols = 200  # 横方向のノード数
n_features = 768  # データの次元数
n_epochs = 1000  # エポック数
learning_rate = 0.1  # 学習率
sigma = 0.5  # 近傍半径の初期値
sigma_decay = sigma / n_epochs  # 減少率


# %%
# コサイン類似度の計算
def calc_cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


# %%
# 近傍関数
def calc_neighborhood(distance, sigma):
    return np.exp(-(distance**2) / (2 * (sigma**2)))


# %%
# マップの作成
def create_torus_som(n_rows, n_cols, n_features):
    som = np.random.rand(n_rows, n_cols, n_features)
    return som


# %%
# BMUの計算
def calc_bmu(som, data):
    bmu = [0, 0]
    max_similarity = 0
    for i in range(n_rows):
        for j in range(n_cols):
            similarity = calc_cosine_similarity(som[i, j], data)
            if similarity > max_similarity:
                max_similarity = similarity
                bmu = [i, j]
    return bmu


# %%
# 近傍ノードの取得
def get_neighborhood(som, bmu, sigma, step):
    x, y = bmu

    # 近傍ノードを格納するリスト
    neighborhood = []
