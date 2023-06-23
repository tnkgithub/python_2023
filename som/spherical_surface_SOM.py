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


#%%
# 測地線ドームを作成
def create_geodesic_dome(num_points, feature_dim):
    vertices = {
        (0.0, 0.0, 1.0): np.random.rand(feature_dim),  # 北極点
        (0.0, 0.0, -1.0): np.random.rand(feature_dim)  # 南極点
    }
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(5):
        latitude = np.arctan(1 / golden_ratio) * i  # 緯度
        longitude_shift = np.pi * i / 5  # 経度シフト

        for j in range(5):
            longitude = 2 * np.pi * j / 5 + longitude_shift
            x = np.cos(longitude) * np.cos(latitude)
            y = np.sin(longitude) * np.cos(latitude)
            z = np.sin(latitude)
            vertices[(x, y, z)] = np.random.rand(feature_dim)

    while len(vertices) < num_points:
        new_vertices = {}

        for i, (current_vertex, _) in enumerate(vertices.items()):
            next_vertex, _ = list(vertices.items())[(i + 1) % len(vertices)]
            midpoint = tuple((np.array(current_vertex) + np.array(next_vertex)) / 2)
            new_data = np.random.rand(feature_dim)
            new_vertices[midpoint] = new_data

        vertices.update(new_vertices)

    return vertices

# 測地線ドームの頂点とデータを生成
num_points = df.shape[0]  # 頂点数を指定してください
feature_dim = df.shape[1]  # データの次元数を指定してください
vertices = create_geodesic_dome(num_points, feature_dim)

# %%
# 球面上の距離を計算
def calc_distance_sphere(v1, v2):
    return np.arccos(np.dot(v1, v2))

#%%
# コサイン類似度の計算
def calc_cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

#%%
# 近傍関数
def calc_neighborhood(distance, sigma):
    return np.exp(-(distance**2) / (2 * (sigma**2)))

#%%
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

#%%
# 近傍ユニットを獲得
def get_neighborhood_unit(bmu, som, sigma):
    # bmuと各頂点の距離を計算し、半径以内の頂点を取得
    neighborhood_unit = []
    for vertex, _ in som.items():
        distance = calc_distance_sphere(bmu, vertex)
        print(distance)
        if distance < sigma:
            neighborhood_unit.append(vertex)
    return neighborhood_unit

#%%
bmu = get_bmu(vertices, df[0])
print(bmu)
#%%
result = get_neighborhood_unit(bmu, vertices, 0.5)
# %%
print(len(result))
# %%
print(calc_distance_sphere((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
# %%
