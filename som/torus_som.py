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

    # BMUから半径以内の座標を取得（距離はユークリッド）
    for i in range(n_rows):
        for j in range(n_cols):
            distance = euclidean([x, y], [i, j])
            if distance <= sigma:
                neighborhood.append([i, j])
            # もし半径以内に端のノードがあった場合、
            # 反対側の端のノードを取得する
            elif (x - sigma < 0) and (i + n_rows - x <= sigma):
