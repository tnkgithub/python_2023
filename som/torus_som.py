# %%
import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

# %%
df = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
df = df.values

# データの次元
dim = df.shape[1]
# データ数
n = df.shape[0]

# マップサイズ
m = 100  # 縦
n = 100  # 横


# %%
# ２点間のユークリッド距離を求める
def euclidean_distance(x, y):
    return euclidean(x, y)


# %%
# コサイン類似度を求める
def cosine_similarity(x, y):
    return 1 - cosine(x, y)


# %%
# コサイン類似度を用いて勝利ノードを求める
def get_winner_node(som, x):
    # 計算結果を入れる空のリストを作成
    tmp = np.arrange(som.shape[0] * som.shape[1]).reshape(som.shape[0], som.shape[1])
    # SOMの各ノードとのコサイン類似度を計算
    for i in range(len(som.shpae[0])):
        for j in range(len(som.shape[1])):
            tmp[i][j] = cosine_similarity(som[i][j], x)
    # コサイン類似度が最大のノードを勝利ノードとする
    return np.unravel_index(np.argmax(tmp), tmp.shape)


# %%
# 近傍ノードの取得（トーラス状）
def get_neighbor_node(som, winner_node, sigma):
    # 近傍ノードを入れる空のリストを作成
    neighbor_node = []
    # 勝利ノードの座標を取得
    x, y = winner_node
    # 近傍ノードの座標を取得
    for i in range(som.shape[0]):
        for j in range(som.shape[1]):
            if euclidean_distance((x, y), (i, j)) < sigma:
                neighbor_node.append((i, j))
    return neighbor_node


# %%
## 画像割り当てアルゴリズムメモ
## １．球面SOMの結果から、画像との類似度がノードを探す
## ２．類似度が高いものがある場合、そのノードに画像を割り当てる
## ３．全画像の類似度を計算し、より類似度が高いものがある場合、更新し、外れた画像を保存する
## ４．外れた画像のみで、３で割り当てられていないノードのみで２．３を繰り返す（再帰）
## ５．全画像が割り当てられたら終了

# %%
