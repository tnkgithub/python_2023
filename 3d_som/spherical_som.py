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

#%%
# データの標準化
scaler = StandardScaler()
df = scaler.fit_transform(df)

df_max = np.max(df)
df_min = np.min(df)

# %%
# 座標データの読み込み
# vertex_df = pd.read_csv("geodesic_dome.csv")
vertex_df = pd.read_csv("../som/result_vertices/torus_34_59.csv")
# vertex_df = pd.read_csv("geocentric_cartesian_coordinates.csv")
vertex_df = vertex_df.values

#%%
print(vertex_df)

# %%
# 辞書{(x, y, z): [feature dim]}を作成
vertices = {}

# dfの長さをマップサイズに合わせる
df_ex = np.zeros((2006, 768))
df_ex[:df.shape[0]] = df

# somで学習するためにデータをシャッフル
np.random.shuffle(df_ex)

for i, vertex in  enumerate(vertex_df):
    #vertices[tuple(vertex)] = (df_max - df_min) * np.random.rand(768) + df_min
    vertices[tuple(vertex)] = df_ex[i]

# %%
# 3次元空間での距離を計算
@jit
def calc_distance(v1, v2):
    diff = np.array(v1) - np.array(v2)
    return np.sqrt(np.dot(diff.T, diff))

# def calc_distance(v1, v2):
#     dist = euclidean(v1, v2)
#     return dist

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
print(bmu)
print(calc_cosine_similarity(vertices[bmu], df[0]))
neighborhood_unit = get_neighborhood_unit(bmu, vertices, 2.5)

print(len(neighborhood_unit))

# %%
# somを学習
som = vertices
n_epochs = 500  # エポック数
learning_rate = 0.5  # 学習率
learning_decay = 0.1  # 学習率の減少率
sigma = 2.5  # 近傍半径の初期値
sigma_decay = 0.1  # 近傍半径の減少率
som = train_som(som, df, n_epochs, learning_rate, learning_decay, sigma, sigma_decay)

# %%
# somの結果をcsvに保存
# 座標のタプルをnumpy配列に変換
vertices_list = []
for vertex, _ in som.items():
    vertices_list.append(np.array(vertex))

now = datetime.datetime.now()
filename = "../som/result_som/som_torus_34_59_" + now.strftime("%Y%m%d_%H%M%S") + ".csv"


# somの結果をデータフレームに変換 x, y, z, feature_dim
df_som = pd.DataFrame(np.concatenate([vertices_list, list(som.values())], axis=1))
df_som.to_csv(filename, index=False)

#%%