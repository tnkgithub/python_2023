# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

# %%
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values

# データの正規化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = scaler.fit_transform(df)


# %%
import numpy as np


def create_geodesic_dome(num_subdivisions):
    # 初期状態の正多面体（正二十面体）の頂点座標を定義
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            [0, 1, phi],
            [0, -1, phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [1, phi, 0],
            [-1, phi, 0],
            [1, -phi, 0],
            [-1, -phi, 0],
            [phi, 0, 1],
            [-phi, 0, 1],
            [phi, 0, -1],
            [-phi, 0, -1],
        ]
    )

    # 多面体の辺を定義
    edges = np.array(
        [
            [0, 8],
            [0, 5],
            [0, 10],
            [0, 11],
            [1, 6],
            [1, 7],
            [1, 8],
            [1, 10],
            [2, 4],
            [2, 5],
            [2, 9],
            [2, 11],
            [3, 6],
            [3, 7],
            [3, 9],
            [3, 11],
            [4, 5],
            [4, 6],
            [4, 9],
            [5, 10],
            [5, 7],
            [6, 8],
            [6, 10],
            [7, 11],
            [8, 10],
            [9, 11],
        ]
    )

    # 繰り返しステップ
    for _ in range(num_subdivisions):
        new_vertices = []
        new_edges = []

        # 各辺の中点を求めて球面に押し出す
        for edge in edges:
            v1 = vertices[edge[0]]
            v2 = vertices[edge[1]]
            mid_point = (v1 + v2) / 2
            normalized_point = mid_point / np.linalg.norm(mid_point)
            new_vertices.append(normalized_point)

        # 新しい頂点を含む新しい辺を構築
        for i in range(len(edges)):
            v1 = edges[i][0]
            v2 = edges[i][1]
            new_vertex_index = len(vertices) + i
            new_edges.append([v1, new_vertex_index])
            new_edges.append([v2, new_vertex_index])
            new_edges.append([new_vertex_index, v1])

        # 頂点と辺を更新
        vertices = np.vstack((vertices, new_vertices))
        edges = np.array(new_edges)

    return vertices


# 細分割の回数を指定して測地線ドームを作成する例
num_subdivisions = 2
dome_vertices = create_geodesic_dome(num_subdivisions)
for vertex in dome_vertices:
    print(vertex)


# %%
fig, ax = plt.subplots(
    figsize=(15, 15), facecolor="white", subplot_kw={"projection": "3d"}
)
X = []
Y = []
Z = []
for vertex in dome_vertices:
    X.append(vertex[0])
    Y.append(vertex[1])
    Z.append(vertex[2])

# ax.plot_wireframe(X, Y, Z, alpha=0.5)  # くり抜き曲面
ax.scatter(X, Y, Z, alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.suptitle("spherical surface", fontsize=10)
ax.set_aspect("auto")
plt.show()


# %%
# 球面上の距離を計算
def calc_distance_sphere(v1, v2):
    return np.arccos(np.dot(v1, v2))


# %%
# コサイン類似度の計算
def calc_cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


# %%
# 近傍関数
def calc_neighborhood(distance, sigma):
    return np.exp(-(distance**2) / (2 * (sigma**2)))


# %%
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


# %%
# 近傍ユニットを獲得
def get_neighborhood_unit(bmu, som, sigma):
    # bmuと各頂点の距離を計算し、半径以内の頂点を取得
    neighborhood_unit = []
    for vertex, _ in som.items():
        distance = calc_distance_sphere(bmu, vertex)
        if distance < sigma:
            neighborhood_unit.append(vertex)
    return neighborhood_unit


# %%
# 重みの更新
def update_weight(som, data, bmu, neighborhood_unit, learning_rate, sigma):
    for vertex, value in som.items():
        if vertex in neighborhood_unit:
            distance = calc_distance_sphere(vertex, bmu)
            neighborhood = calc_neighborhood(distance, sigma)
            som[vertex] = value + learning_rate * neighborhood * (data - value)
    return som


# %%
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
# 測地線ドームの頂点とデータを生成
num_points = 4000  # 頂点数を指定してください
feature_dim = df.shape[1]  # データの次元数を指定してください
som = create_geodesic_dome(num_points, feature_dim)

# %%
# somを学習
n_epochs = 500  # エポック数を指定してください
learning_rate = 0.5  # 学習率を指定してください
learning_decay = 0.1  # 学習率の減少率を指定してください
sigma = 0.5  # 近傍半径の初期値を指定してください
sigma_decay = 0.1  # 近傍半径の減少率を指定してください
som = train_som(som, df, n_epochs, learning_rate, learning_decay, sigma, sigma_decay)

# %%
# 座標を取得 [x, y, z]
coordinates = []
for vertex, _ in som.items():
    coordinates.append(vertex)

coordinates = np.array(coordinates)

# %%
print(coordinates[:, 0])

# %%
# ３次元にプロット
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
plt.show()


# %%
df = pd.DataFrame.from_dict(som, orient="index")
df.to_csv("som.csv")

# %%
df = pd.read_csv("som.csv", index_col=None)
df_dict = df.to_dict(orient="index")

# %%
print(df_dict.keys())

# %%
# 3次元の座標を2次元に変換
som3d_map = []
for vertex, _ in som.items():
    tmp = np.array(vertex)
    som3d_map.append(tmp)

som3d_map = np.array(som3d_map)
# %%


# %%
