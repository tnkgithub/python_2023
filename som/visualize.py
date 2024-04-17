#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import datetime
import csv

#%%
# 特徴量の読み込み
df_features = pd.read_csv("../ViT/result_feature/feature_vit.csv", index_col=0)
features = df_features.values


# som結果の読み込み
df_som = pd.read_csv("./result_som/som_torus_43_46_20230922_213922.csv", index_col=0)
vertices = df_som.iloc[:, :3].values
som = df_som.iloc[:, 2:].values

m = 34 # 軸周りの分割数
n = 59 # 断面円の分割数

#%%
# cos類似度の計算
def calc_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#%%
# アルゴリズム
# 1. som結果と特徴量の類似度を計算
# 2. 類似度が高い箇所に特徴量を配置
# 3. もし、2ですでに特徴量が配置されていたら、類似度が高い方に書き換える。
# 4. 1~4を繰り返し、全ての特徴量が配置されるまで続ける。

# 2次元配列に変換
flat = [[-1] * m for _ in range(n)]
# 特徴量のインデックス
features_index = [i for i in range(len(features))]
# どこにも配置されていない特徴量のインデックス
not_placed_features_index = []

def som_to_gird():

    for i in features_index[:]:
        # 1. som結果と特徴量の類似度を計算
        similarity = [calc_cosine_similarity(features[i], som[j]) for j in range(len(som))]
        # 2. 類似度が高い箇所に特徴量を配置
        max_similarity = max(similarity)
        max_index = similarity.index(max_similarity)
        # 3. もし、2ですでに特徴量が配置されていたら、類似度が高い方に書き換える。
        if flat[max_index // m][max_index % m] != -1:
            # 類似度が高い方に書き換える
            if max_similarity > calc_cosine_similarity(features[flat[max_index // m][max_index % m]], som[max_index]):
                features_index.append(flat[max_index // m][max_index % m])
                flat[max_index // m][max_index % m] = i
                features_index.remove(i)
            else:
                if i not in not_placed_features_index:
                    not_placed_features_index.append(i)
                features_index.remove(i)
        else:
            flat[max_index // m][max_index % m] = i
            features_index.remove(i)


    print(len(features_index))
    if len(features_index) != 0:
        som_to_gird()

    print(len(not_placed_features_index))
    # まだ配置されていない特徴量がある場合

#%%
def som_to_gird2(max_similarity=0.3):
    tmp = len(not_placed_features_index)
    if len(not_placed_features_index) != 0:
        # girdの配されていない箇所のうち、最も類似度が高いものに配置する
        for i in not_placed_features_index:
            max_x = 1000
            max_y = 1000
            for x in range(n):
                for y in range(m):
                    if flat[x][y] == -1:
                        similarity = calc_cosine_similarity(features[i], som[x * m + y])
                        if max_similarity < similarity:
                            max_similarity = similarity
                            max_x = x
                            max_y = y

            if max_x != 1000 and max_y != 1000:
                flat[max_x][max_y] = i
                not_placed_features_index.remove(i)

    if tmp == len(not_placed_features_index):
        return
    else:
        max_similarity = max_similarity - 0.1
        som_to_gird2(max_similarity)

# %%
som_to_gird()
#%%
som_to_gird2(0.3)

#%%
#１行にする？
img_no = []
img_no = sum(flat, [])

now = datetime.datetime.now()
filename = './result_som_list/result_SOM_torus_50_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'

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
img_name = './result_image/result_SOM_image_torus_43_46_' + now.strftime('%Y%m%d_%H%M%S') + '.png'

img_dir_path =   '../scraping/images/'

#%%

# 画像のパスを取得
imgs_path = []
img_list = os.listdir(img_dir_path)
for i in img_list:
    imgs_path.append(img_dir_path + i)


# 画像のサイズ、背景を設定
plt.figure(figsize=(50,50), facecolor='w')
plt.subplots_adjust(wspace=0, hspace=0)

'''画像を読み込み、タイル状に出力'''
imgs = [0] * m * n
for i, no in zip(range(m*n), img_no):
    if no != -1:
        # 画像を読み込む
        imgs[i] = cv2.imread(imgs_path[no])
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        #imgs[i] = resize_and_trim(imgs[i], 100, 162)
        plt.subplot(m, n, i+1)
        plt.subplots_adjust(hspace=0.0)
        plt.axis("off")
        plt.imshow(imgs[i])


plt.savefig(img_name)


# %%
img_name = '../som/digital_archive_image.png'
imgs_dir_path =   '../scraping/images/'

img_list = pd.read_csv('../som/image_som_result20230110_073816.csv', index_col=0)
img_list = img_list.index.tolist()

#%%
print(len(img_list))

#%%
m = 28
n = 66

# 画像のサイズ、背景を設定
plt.figure(figsize=(50,35), facecolor='w')
plt.subplots_adjust(wspace=0, hspace=0)

'''画像を読み込み、タイル状に出力'''
imgs = [0] * m * n
for i, no in zip(range(m*n), img_list):
    if no != '1':
        # 画像を読み込む
        imgs[i] = cv2.imread(imgs_dir_path + no[:8] + '.jpg')
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        #imgs[i] = resize_and_trim(imgs[i], 100, 162)
        plt.subplot(m, n, i+1)
        plt.subplots_adjust(hspace=0.0)
        plt.axis("off")
        plt.imshow(imgs[i])


plt.savefig(img_name)

# %%
