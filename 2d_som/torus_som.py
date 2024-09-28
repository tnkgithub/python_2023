#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import datetime
import csv

#%%
n = 78 #(横)
m = 26 #(縦)
np.random.seed(1)

#%%
df = pd.read_csv('/home/b1019035/2023/python_2023/ViT/result_feature/feature_vit_normal_HF.csv', index_col=0)
features = df.to_numpy()
features = features.astype(np.float32)
print(features)

#%%
# 正規化
features = features / np.linalg.norm(features, axis=1).reshape(-1, 1)

# %%
def cos_similarity(v1, v2):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0
    return np.dot(v1, v2) / (v1_norm * v2_norm)


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def wrap_around(x, max_val):
    """
    トーラス型のためのラップアラウンド関数
    """
    return (x + max_val) % max_val

def find_bmu(som, ex_data):
    # 計算結果を格納する配列
    tmp = np.arange(som.shape[0] * som.shape[1]).reshape(som.shape[0], som.shape[1])
    # 類似度計算
    for x in range(som.shape[0]):
        for y in range(som.shape[1]):
            # tmp[x, y] = cos_similarity(som[x, y], ex_data)
            tmp[x, y] = euclidean_distance(som[x, y], ex_data)

    # 類似度が最大(距離が最小)のユニットの座標を返す
    return np.unravel_index(np.argmin(tmp, axis=None), tmp.shape)
    # return np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

def update_weight(som, train_ex_data, learn_rate, radius_sq, bmu_coord, step=3):
    # BMUの座標
    g, h = bmu_coord

    #
    if radius_sq < 1e-3:
        som[g, h, :] += learn_rate * (train_ex_data - som[g, h, :])
        return som

    for i in range(g-step, g+step+1):
        for j in range(h-step, h+step+1):
            # トーラス型のためのラップアラウンド
            wrapped_i = wrap_around(i, som.shape[0])
            wrapped_j = wrap_around(j, som.shape[1])

            # ラップアラウンド後の距離の二乗
            dist_i = min(abs(wrapped_i - g), som.shape[0] - abs(wrapped_i - g))
            dist_j = min(abs(wrapped_j - h), som.shape[1] - abs(wrapped_j - h))
            dist_sq = np.square(dist_i) + np.square(dist_j)

            # 近傍関数
            dist_func = np.exp(-dist_sq / (2 * radius_sq))
            som[wrapped_i, wrapped_j, :] += learn_rate * dist_func * (train_ex_data - som[wrapped_i, wrapped_j, :])

    return som
# %%
def train_som(som, train_data, learn_rate=0.1, radius_sq=200, lr_decay=.1, rad_decay=.1, epochs=10):
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
# som = np.random.rand(m, n, features.shape[1])
# 初期値を特徴量で初期化
index = np.arange(features.shape[0])
# np.random.shuffle(index)
som = np.zeros((m, n, features.shape[1])).astype(np.float32)
for i in range(m):
    for j in range(n):
        if i*n+j >= features.shape[0]:
            som[i, j] = np.zeros(features.shape[1])
        else:
            som[i, j] = features[index[i*n+j]]

print(som.shape)
print(features.shape)
# %%
# 学習
# featuresの順番をシャッフル
random_features = features.copy()
np.random.shuffle(random_features)

som = train_som(som, random_features, learn_rate=0.5, radius_sq=100, lr_decay=0.01, rad_decay=0.01, epochs=200)


# %%
# somを一次元に変換
som_1d = som.reshape(m*n, features.shape[1])
print(som_1d)

#%%
# 学習結果を保存
now = datetime.datetime.now()
os.makedirs('../2d_som/result_weight', exist_ok=True)

df = pd.DataFrame(som_1d)
df.to_csv(f"../2d_som/result_weight/som_weight_{now:%Y%m%d_%H%M%S}.csv", index=False)

#%%

# # 学習結果の可視化
# # 画像を配置するための2次元座標を作成



# # どこにも配置されていない特徴量のインデックス
# features_index = [i for i in range(features.shape[0])]

# # まだ配置されていない特徴量のインデックス
# not_placed_features_index = []
# placed_features_index = []

# def som_to_gird():

#     for i in range(2003):
#         print("i", i)
#         # 1. som結果と特徴量の類似度を計算
#         similarity = [cos_similarity(features[i], som_1d[j]) for j in range(len(som_1d))]
#         #similarity = [dis.euclidean(features[i], som_1d[j]) for j in range(len(som_1d))]

#         max_similarity = max(similarity)
#         max_index = similarity.index(max_similarity)
#         # 3. もし、2ですでに特徴量が配置されていたら、類似度が高い方に書き換える。
#         if map[max_index // m][max_index % m] != -1:
#             if max_similarity > cos_similarity(features[map[max_index // m][max_index % m]], som_1d[max_index]):
#                 placed_features_index.append(i)
#                 placed_features_index.remove(map[max_index // m][max_index % m])
#                 not_placed_features_index.append(map[max_index // m][max_index % m])
#                 map[max_index // m][max_index % m] = i
#                 print("replace", max_index // m, max_index % m)
#             else:
#                 print("still exist", max_index // m, max_index % m)
#                 not_placed_features_index.append(i)
#         # 2. 類似度が高い箇所に特徴量を配置
#         else:
#             map[max_index // m][max_index % m] = i
#             placed_features_index.append(i)
#             print("new", max_index // m, max_index % m)

#     print("not_placed_features_index_len", len(not_placed_features_index))
#     print(not_placed_features_index)

# #%%
# def som_to_gird2(max_similarity=0.3):
#     print(max_similarity)
#     tmp_similarity = max_similarity
#     if len(not_placed_features_index) != 0:
#         # girdの配されていない箇所のうち、最も類似度が高いものに配置する
#         for i in not_placed_features_index:
#             max_x = 1000
#             max_y = 1000
#             max_similarity = tmp_similarity
#             for x in range(n):
#                 for y in range(m):
#                     if map[x][y] == -1:
#                         similarity = cos_similarity(features[i], som_1d[x * m + y])
#                         #similarity = dis.euclidean(features[i], som_1d[x * m + y])
#                         if max_similarity < similarity:
#                             max_similarity = similarity
#                             max_x = x
#                             max_y = y

#             if max_x != 1000 and max_y != 1000:
#                 print("new", max_x, max_y, "i", i)
#                 map[max_x][max_y] = i
#                 not_placed_features_index.remove(i)

#         print(len(not_placed_features_index))
#         max_similarity = round(tmp_similarity - 0.1, 2)
#         som_to_gird2(max_similarity)

#     else:
#         return

# som_to_gird()
# som_to_gird2(0.3)


#%%
map = [[-1] * m for _ in range(n)]
not_placed_features_index = []
index_list = [i for i in range(features.shape[0])]

def som_grid(max_similarity=1.0, index_list=[]):
    tmp_similarity = max_similarity
    for i in index_list:
        max_x = 1000
        max_y = 1000
        for x in range(n):
            for y in range(m):
                similarity = cos_similarity(features[i], som_1d[x * m + y])
                if max_similarity <= similarity:
                    if map[x][y] != -1:
                        print("still exist", x, y)
                        continue
                    max_similarity = similarity
                    max_x = x
                    max_y = y


        if max_x != 1000 and max_y != 1000:
            print(f"new x: {max_x}, y: {max_y}, i: {i}, sim: {max_similarity}")
            map[max_x][max_y] = i
            if i in not_placed_features_index:
                not_placed_features_index.remove(i)
        else:
            if i not in not_placed_features_index:
                print(f"not placed i: {i}")
                not_placed_features_index.append(i)
        max_similarity = tmp_similarity

    if len(not_placed_features_index) != 0:
        max_similarity = round(tmp_similarity - 0.1, 2)
        print(f"not placed features: {len(not_placed_features_index)}")
        som_grid(max_similarity, not_placed_features_index)
    else:
        return


som_grid(1.0, index_list)


#%%
# 画像の配置結果を保存
now = datetime.datetime.now()
os.makedirs('../2d_som/result_map', exist_ok=True)

#
img_no = []
img_no = sum(map, [])

now = datetime.datetime.now()
filename = f"../2d_som/result_map/result_{now:%Y%m%d_%H%M%S}.csv"

f = open(filename, 'w')
writer = csv.writer(f)
writer.writerow(img_no)
f.close()

#%%
def resize_and_trim(img, width, height):
    h, w = img.shape[:2]
    magnification = height / h
    reWidth = int(magnification * w)
    size = (reWidth, height)

    if reWidth < 100:
        magnification = width / w
        reHeight = int(magnification * h)
        size = (width, reHeight)

    img_resize = cv2.resize(img, size)

    h, w = img_resize.shape[:2]

    top = int((h / 2) - (height / 2))
    bottom = top+height
    left = int((w / 2) - (width / 2))
    right = left+width

    return img_resize[top:bottom, left:right]

#%%
# 画像の配置結果を画像として保存
result_img_name = '../2d_som/result_som_image/result_SOM_image_torus_34_59_' + now.strftime('%Y%m%d_%H%M%S') + '.png'
img_dir_path = '/home/b1019035/dev/theme-app/web/public/posters/normalPoster/'

# 画像の読み込み
imgs_path = []
img_list = os.listdir(img_dir_path)
for i in img_list:
    imgs_path.append(img_dir_path + i)

print(imgs_path)
print(len(imgs_path))
#%%

# 画像の配置
# 画像のサイズ、背景を設定
plt.figure(figsize=(50,35), facecolor='w')
plt.subplots_adjust(wspace=0, hspace=0)

'''画像を読み込み、タイル状に出力'''
imgs = [0] * m * n
for i, no in zip(range(m*n), img_no):
    if no != -1:
        # 画像を読み込む
        imgs[i] = cv2.imread(imgs_path[no])
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        imgs[i] = resize_and_trim(imgs[i], 100, 162)
        plt.subplot(m, n, i+1)
        plt.subplots_adjust(hspace=0.0)
        plt.axis("off")
        plt.imshow(imgs[i])


plt.savefig(result_img_name)
# %%
