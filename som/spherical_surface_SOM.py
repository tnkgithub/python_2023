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
# 1. 初期状態として球に内接する正２０面体を作成
# 2. 多面体の各辺の中点をとる
# 3. 各辺の中点を球面に接するように移動させる
# 4. 移動した点を新たな頂点として多面体を作成
# 5. 2. 3. 4. を繰り返し、頂点がデータ数より多くなったら終了

import math


def create_geodesic_dome(data, num_iterations):
    # 初期状態として球に内接する正20面体を作成
    vertices = create_icosahedron()

    for _ in range(num_iterations):
        # 多面体の各辺の中点を取得
        midpoints = get_edge_midpoints(vertices)

        # 各辺の中点を球面に接するように移動
        vertices = move_points_to_sphere(midpoints)

        # 移動した点を新たな頂点として多面体を作成
        vertices = create_polyhedron(vertices)

        # 頂点数がデータ数を超えたら終了
        if len(vertices) > len(data):
            break

    # データを頂点に格納
    for i, point in enumerate(data):
        if i >= len(vertices):
            break
        vertices[i] = point

    return vertices


def create_icosahedron():
    phi = (1 + math.sqrt(5)) / 2  # 黄金比
    vertices = [
        (0, +1, +phi),
        (0, +1, -phi),
        (0, -1, +phi),
        (0, -1, -phi),
        (+1, +phi, 0),
        (+1, -phi, 0),
        (-1, +phi, 0),
        (-1, -phi, 0),
        (+phi, 0, +1),
        (+phi, 0, -1),
        (-phi, 0, +1),
        (-phi, 0, -1),
    ]
    return vertices


def get_edge_midpoints(vertices):
    midpoints = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        vertex1 = vertices[i]
        vertex2 = vertices[(i + 1) % num_vertices]
        midpoint = (
            (vertex1[0] + vertex2[0]) / 2,
            (vertex1[1] + vertex2[1]) / 2,
            (vertex1[2] + vertex2[2]) / 2,
        )
        midpoints.append(midpoint)

    return midpoints


def move_points_to_sphere(points):
    radius = 1.0  # 半径1の球
    normalized_points = []

    for point in points:
        x, y, z = point
        length = math.sqrt(x**2 + y**2 + z**2)

        # ゼロ除算を防止
        if length != 0:
            normalized_x = radius * (x / length)
            normalized_y = radius * (y / length)
            normalized_z = radius * (z / length)
        else:
            normalized_x, normalized_y, normalized_z = x, y, z

        normalized_points.append((normalized_x, normalized_y, normalized_z))

    return normalized_points


def create_polyhedron(vertices):
    new_vertices = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        vertex1 = vertices[i]
        vertex2 = vertices[(i + 1) % num_vertices]
        new_vertex = (
            (vertex1[0] + vertex2[0]) / 2,
            (vertex1[1] + vertex2[1]) / 2,
            (vertex1[2] + vertex2[2]) / 2,
        )
        new_vertices.append(new_vertex)

    return new_vertices


# 使用例
num_iterations = 5

result = create_geodesic_dome(df, num_iterations)
result = np.array(result)
print(result.shape)

# %%
