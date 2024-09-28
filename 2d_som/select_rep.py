# 代表資料を選定する
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

#%%
# データの読み込み
with open("/home/sumthen/2023/python_2023/2d_som/result_map/result_20240923_060625.csv", "r") as f:
    reader = csv.reader(f)
    map_data = [row for row in reader]

# map_dataを１次元に変換
map_data = np.array(map_data).flatten()

print(map_data)
# %%
# 特徴量の読み込み
df = pd.read_csv("/home/sumthen/2023/python_2023/ViT/result_feature/feature_vit_normal_HF.csv", index_col=0,)
df_index = list(df.index)
# dfをnp.arrayに変換(float64)
features = df.values.astype(np.float64)


# %%
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# %%
# map_dataを縦26×横78に変換
map_data_grid = np.array(map_data).reshape(26, 78)

# 代表データの選定
# 3×3の範囲で特徴量の平均を算出し、その範囲内で最も近い代表データを選定
representative_data = []
for i in range(1, 26, 3):
    for j in range(1, 78, 3):
        feature = np.zeros(768)
        if i != 25:
            for x in range(i-1, i+2):
                for y in range(j-1, j+2):
                    feature += features[int(map_data_grid[x, y])]
            average_feature = feature / 9
            max_similarity = 0
            max_index = -1
            for x in range(i-1, i+2):
                for y in range(j-1, j+2):
                    similarity = cosine_similarity(average_feature, features[int(map_data_grid[x, y])])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_index = int(map_data_grid[x, y])
            representative_data.append(max_index)
            print(max_index)
        else:
            for x in range(i-1, i+1):
                for y in range(j-1, j+2):
                    feature += features[int(map_data_grid[x, y])]
            average_feature = feature / 6
            max_similarity = 0
            max_index = -1
            for x in range(i-1, i+1):
                for y in range(j-1, j+2):
                    similarity = cosine_similarity(average_feature, features[int(map_data_grid[x, y])])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_index = int(map_data_grid[x, y])
            representative_data.append(max_index)
            print(max_index)
#%%
print(len(representative_data))

#%%
# 辞書にしてcsvに保存
representative_dict = {df_index[representative_data[i]]: i for i in range(len(representative_data))}
with open("/home/sumthen/2023/python_2023/2d_som/result_map/representative_number.csv", "w") as f:
    writer = csv.writer(f)
    for key, value in representative_dict.items():
        writer.writerow([key, value])
# %%

for i in range(1, 26, 3):
    for j in range(1, 78, 3):
        print(i, j)
# %%
