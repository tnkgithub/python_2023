#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

#%%
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values

# データの正規化
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

#%%
# 半径を指定
r = 1.0

# ラジアンを作成
# q: 0.0 ~ 2.0*np.piとは何か
# a: 0.0 ~ 2.0*np.piの範囲で81個の等間隔の数値を生成
t = np.linspace(start=0.0, stop=2.0*np.pi, num=81)
u = np.linspace(start=0.0, stop=2.0*np.pi, num=81)

# 格子点を作成
T, U = np.meshgrid(t, u)

# 球面座標を直交座標に変換
X = r * np.sin(T) * np.cos(U)
Y = r * np.sin(T) * np.sin(U)
Z = r * np.cos(T)

#%%
XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

# %%
# # 球面を作図
# fig, ax = plt.subplots(figsize=(15, 15), facecolor='white', subplot_kw={'projection': '3d'})
# ax.scatter(X, Y, Z, c=Z, cmap='viridis', alpha=0.5) # 散布図:(z軸の値により色付け)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# fig.suptitle('spherical surface', fontsize=20)
# ax.set_aspect('auto')
# plt.show()

# %%
# 辞書{(x, y, z): [feature_dim]}を作成
vertices = {}
for i in range(len(XYZ)):
    vertices[tuple(XYZ[i])] = np.random.rand(768)

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
# somを学習
som = vertices
n_epochs = 100  # エポック数を指定してください
learning_rate = 0.5  # 学習率を指定してください
learning_decay = 0.1  # 学習率の減少率を指定してください
sigma = 1.0  # 近傍半径の初期値を指定してください
sigma_decay = 0.1  # 近傍半径の減少率を指定してください
som = train_som(som, df, n_epochs, learning_rate, learning_decay, sigma, sigma_decay)

# %%
# somの結果をcsvに保存
# 座標のタプルをnumpy配列に変換
vertices_list = []
for vertex, _ in som.items():
    vertices_list.append(np.array(vertex))

#%%
print(vertices_list)
#%%

# somの結果をデータフレームに変換 x, y, z, feature_dim
df_som = pd.DataFrame(np.concatenate([vertices_list, list(som.values())], axis=1))
df_som.to_csv('som.csv', index=False)

# %%
som_vertices = df_som.iloc[:, :3].values
som_value = df_som.iloc[:, 3:].values

#%%
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
        y = np.log(np.tan((lat + 90) * np.pi / 360 + np.pi / 4)) * (height / (2 * np.pi))
        xy.append([x, y])
    return np.array(xy)

# %%
xy = convert_latlon_to_xy(latlon)

#%%
# 地図の描画
xy = convert_latlon_to_xy(latlon)
width = 100
height = 100
x = xy[:, 0]
y = xy[:, 1]
plt.plot(x, y, 'ro')
plt.xlim(0, width)
plt.ylim(0, height)
plt.show()
# %%
