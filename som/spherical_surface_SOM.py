# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values


# %%
# 測地ドームの作成
import math
import numpy as np


def create_geodesic_dome(num_points):
    vertices = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])  # 北極点  # 南極点
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(5):
        latitude = np.arctan(1 / golden_ratio) * i  # 緯度
        longitude_shift = np.pi * i / 5  # 経度シフト

        for j in range(5):
            longitude = 2 * np.pi * j / 5 + longitude_shift
            x = np.cos(longitude) * np.cos(latitude)
            y = np.sin(longitude) * np.cos(latitude)
            z = np.sin(latitude)
            vertices = np.vstack((vertices, [x, y, z]))

    while len(vertices) < num_points:
        new_vertices = []

        for i in range(len(vertices)):
            current_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % len(vertices)]
            midpoint = (current_vertex + next_vertex) / 2
            new_vertices.append(midpoint)

        vertices = np.vstack((vertices, new_vertices))

    return vertices


# 測地線ドームの頂点を生成
num_points = df.shape[0]
vertices = create_geodesic_dome(num_points)

# %%
print(vertices)

som = np.expand_dims(vertices, axis=2)

# %%
print(som)

# %%
# マップの作成(3次元目に768次元のデータをランダムに割り当てる)


# %%
print(som.shape)

# %%
