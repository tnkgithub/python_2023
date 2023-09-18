# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from numba import jit

# 標準化
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean

# %%
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values

# データの標準化
scaler = StandardScaler()
df = scaler.fit_transform(df)

df_max = np.max(df)
df_min = np.min(df)

# %%
# 座標データの読み込み
# vertex_df = pd.read_csv("geodesic_dome.csv")
vertex_df = pd.read_csv("result_vertices/torus_43_46.csv")
# vertex_df = pd.read_csv("geocentric_cartesian_coordinates.csv")
vertex_df = vertex_df.values

# %%
# 辞書{(x, y, z): [feature dim]}を作成
vertices = {}
for vertex in vertex_df:
    vertices[tuple(vertex)] = (df_max - df_min) * np.random.rand(768) + df_min

# %%
tmp = (df_max - df_min) * np.random.rand(768) + df_min
print(tmp.shape)
print(tmp.max())
print(tmp.min())


# %%
# 3次元空間での距離を計算
@jit
def calc_distance(v1, v2):
    diff = np.array(v1) - np.array(v2)
    return np.sqrt(np.dot(diff.T, diff))

# コサイン類似度の計算
def calc_cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


# 近傍関数
def calc_neighborhood(distance, sigma):
    return np.exp(-(distance**2) / (2 * (sigma**2)))

# bmuを獲得
def get_bmu(som, data):
    max_similarity = 0
    bmu = 0
    for vertex, value in som.items():
        similarity = calc_cosine_similarity(value, data)
        if similarity > max_similarity:
            max_similarity = similarity
            bmu = vertex
    return bmu


# 近傍ユニットを獲得
def get_neighborhood_unit(bmu, som, sigma):
    # bmuと各頂点の距離を計算し、半径以内の頂点を取得
    neighborhood_unit = []
    for vertex, _ in som.items():
        distance = calc_distance(bmu, vertex)
        if distance < sigma:
            neighborhood_unit.append(vertex)
    return neighborhood_unit

# 重みの更新
def update_weight(som, data, bmu, neighborhood_unit, learning_rate, sigma):
    for vertex, value in som.items():
        if vertex in neighborhood_unit:
            distance = calc_distance(vertex, bmu)
            neighborhood = calc_neighborhood(distance, sigma)
            som[vertex] = value + learning_rate * neighborhood * (data - value)
    return som


# somを学習
def train_som(som, data, n_epochs, learning_rate, learning_decay, sigma, sigma_decay):
    learning_rate_0 = learning_rate
    sigma_0 = sigma
    for epoch in range(n_epochs):
        for i in range(data.shape[0]):
            bmu = get_bmu(som, data[i])
            neighborhood_unit = get_neighborhood_unit(bmu, som, sigma)
            som = update_weight(
                som, data[i], bmu, neighborhood_unit, learning_rate, sigma
            )
        learning_rate = learning_rate_0 * np.exp(-epoch / learning_decay)
        sigma = sigma_0 * np.exp(-epoch / sigma_decay)
    return som


# %%
bmu = get_bmu(vertices, df[0])
neighborhood_unit = get_neighborhood_unit(bmu, vertices, 0.35)

print(len(neighborhood_unit))

# %%
# somを学習
som = vertices
n_epochs = 1500  # エポック数
learning_rate = 0.5  # 学習率
learning_decay = 0.1  # 学習率の減少率
sigma = 0.35  # 近傍半径の初期値
sigma_decay = 0.1  # 近傍半径の減少率
som = train_som(som, df, n_epochs, learning_rate, learning_decay, sigma, sigma_decay)

# %%
# somの結果をcsvに保存
# 座標のタプルをnumpy配列に変換
vertices_list = []
for vertex, _ in som.items():
    vertices_list.append(np.array(vertex))

# %%
now = datetime.datetime.now()
filename = "./result_som/som_torus_43_46_" + now.strftime("%Y%m%d_%H%M%S") + ".csv"


# somの結果をデータフレームに変換 x, y, z, feature_dim
df_som = pd.DataFrame(np.concatenate([vertices_list, list(som.values())], axis=1))
df_som.to_csv(filename, index=False)

#
# ここまで
#

# %%
df_som = pd.read_csv("som.csv")
som_vertices = df_som.iloc[:, :3].values
som_value = df_som.iloc[:, 3:].values

# %%
# 球面を作図
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="white", subplot_kw={"projection": "3d"}
)
X = som_vertices[:, 0]
Y = som_vertices[:, 1]
Z = som_vertices[:, 2]
ax.scatter(X, Y, Z, c=Z, cmap="viridis", alpha=0.5)  # 散布図:(z軸の値により色付け)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.suptitle("spherical surface", fontsize=20)
ax.set_aspect("auto")
plt.show()


# %%
# 地心直交座標を緯度経度に変換
def convert_xyz_to_latlon(vertices):
    latlon = []
    for vertex in vertices:
        x, y, z = vertex
        lon = np.arctan2(y, x)
        hyp = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, hyp)
        lat = np.degrees(lat)
        lon = np.degrees(lon)
        latlon.append([lat, lon])
    return np.array(latlon)


# %%
latlon = convert_xyz_to_latlon(som_vertices)


# %%
# 緯度経度から平面直角座標に変換
def convert_latlon_to_xy(latlon, width=100, height=100):
    xy = []
    for lat, lon in latlon:
        x = (lon + 180) * (width / 360)
        y = np.log(np.tan((lat + 90) * np.pi / 360 + np.pi / 4)) * (
            height / (2 * np.pi)
        )
        xy.append([x, y])
    return np.array(xy)


# %%
xy = convert_latlon_to_xy(latlon)

# %%
# 地図の描画
xy = convert_latlon_to_xy(latlon)
width = 100
height = 100
x = xy[:, 0]
y = xy[:, 1]
plt.plot(x, y, "ro")
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()
# %%
