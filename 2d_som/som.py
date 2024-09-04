#%%
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
from scipy.spatial.distance import cosine, euclidean

#%%
m = 59 #(横)
n = 34 #(縦)
np.random.seed(1)

#%%
df = pd.read_csv('/home/b1019035/2023/python_2023/ViT/result/result_feature/feature_vit.csv', index_col=0)
features = df.to_numpy()

features = features.astype(np.float32)
print(features)

#%%
def cos_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def euclidean_distance(v1, v2):
    return euclidean(v1, v2)

def find_bmu(som, ex_data):
    # 計算結果を格納する配列
    tmp = np.arange(som.shape[0] * som.shape[1]).reshape(som.shape[0], som.shape[1])
    # 類似度計算
    for x in range(som.shape[0]):
        for y in range(som.shape[1]):
            #tmp[x, y] = cos_similarity(som[x, y], ex_data)
            tmp[x, y] = euclidean(som[x, y], ex_data)

    # 類似度が最大のユニットの座標を返す
    return np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

def update_weight(som, train_ex_data, learn_rate, radius_sq, bmu_coord, step=3):
    # BMUの座標
    g, h = bmu_coord

    #
    if radius_sq < 1e-3:
        som[g, h, :] += learn_rate * (train_ex_data - som[g, h, :])
        return som

    for i in range(max(0, g - step), min(g + step, som.shape[0])):
        for j in range(max(0, h - step), min(h + step, som.shape[1])):
            # ユニットとBMUのユークリッド距離の二乗
            dist_sq = np.square(i-g) + np.square(j-h)
            # 近傍関数
            dist_func = np.exp(-dist_sq / (2 * radius_sq))
            # 重みの更新
            som[i, j, :] += learn_rate * dist_func * (train_ex_data - som[i, j, :])

    return som

def train_som(som, train_data, learn_rate=0.1, radius_sq=10, lr_decay=.1, rad_decay=.1, epochs=10):
    # 学習率の初期値
    learn_rate_0 = learn_rate
    # 近傍半径の初期値
    radius_sq_0 = radius_sq

    # 学習
    for epoch in range(epochs):
        for ex_date in train_data:
            g, h = find_bmu(som, ex_date)
            som = update_weight(som, ex_date, learn_rate, radius_sq, (g, h))

        # 学習率と近傍半径の減衰
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_sq_0 * np.exp(-epoch * rad_decay)

    return som


#%%
# SOMの初期化
som = np.random.rand(n, m, features.shape[1])

# %%
som = train_som(som, features, learn_rate=0.5, lr_decay=0.1, radius_sq=2500, rad_decay=0.001, epochs=5000)

#%%
# 画像の生成
list_som = som.reshape(m*n, features.shape[1])
map = [-1] * (m*n)
features_index = [i for i in range(features.shape[0])]


# どこにも配置されていない特徴量のインデックス
not_placed_features_index = []


#%%
def som_to_map(list):

    for i in list:
        similarity = [cos_similarity(features[i], list_som[j]) for j in range(len(list_som))]
        #類似度とインデックスの２次元配列
        similarity_index = [[similarity[j], j] for j in range(len(list_som))]
        sorted_similarity_index = sorted(similarity_index, reverse=True)
        # 2. 類似度が高い箇所に特徴量を配置
        for sim, index in sorted_similarity_index:
            if map[index] == -1:
                map[index] = i
                if i in not_placed_features_index:
                    not_placed_features_index.remove(i)
                break
            else:
                if cos_similarity(features[map[index]], list_som[index]) < sim:
                    not_placed_features_index.append(map[index])
                    if i in not_placed_features_index:
                        not_placed_features_index.remove(i)
                    map[index] = i
                    break
                else:
                    continue

    if len(not_placed_features_index) != 0:
        print(len(not_placed_features_index))
        som_to_map(not_placed_features_index)

#%%
som_to_map(features_index)




#%%
now = datetime.datetime.now()

#%%
os.makedirs('../2d_som/result_som_csv/', exist_ok=True)

filename = '../2d_som/result_som_csv/result_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'

for i in map:
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i])


#%%
os.makedirs('../2d_som/result_som_image', exist_ok=True)

img_name = '../2d_som/result_som_image/result_' + now.strftime('%Y%m%d_%H%M%S') + '.png'

img_dir_path =   '../scraping/images/'

# 画像のパスを取得
imgs_path = []
img_list = df.index.tolist()
for i in img_list:
    imgs_path.append(img_dir_path + i + '.jpg')

print(imgs_path)

#%%


# 画像のサイズ、背景を設定
plt.figure(figsize=(50,50), facecolor='w')
plt.subplots_adjust(wspace=0, hspace=0)

'''画像を読み込み、タイル状に出力'''
imgs = [0] * m * n
for i, no in zip(range(m*n), map):
    if no != -1:
        # 画像を読み込む
        imgs[i] = cv2.imread(imgs_path[no])
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        #imgs[i] = resize_and_trim(imgs[i], 100, 162)
        plt.subplot(n, m, i+1)
        plt.subplots_adjust(hspace=0.0)
        plt.axis("off")
        plt.imshow(imgs[i])


plt.savefig(img_name)

# %%
