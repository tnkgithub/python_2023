#%%
import pandas as pd
import numpy as np
import os
import csv

#%%
# mapの読み込み
with open("/home/sumthen/2023/python_2023/2d_som/result_map/result_20240923_060625.csv", "r") as f:
    reader = csv.reader(f)
    map_data = [row for row in reader]
# map_dataを１次元に変換
map_data = np.array(map_data).flatten()
# map_dataをリストに変換
map_data = map_data.tolist()

# 特徴量の読み込み
df = pd.read_csv("/home/sumthen/2023/python_2023/ViT/result_feature/feature_vit_normal_HF.csv", index_col=0)
df_index = list(df.index)

#%%
# po000001.jpgがdf_indexの何番目にあるかを調べる
map_index = {df_index[i]: i for i in range(len(df_index))}

# %%
# {df_index[map_data[i], i}を辞書にしておく
index_dict = {}
gap_counter = 0
for i in range(len(map_data)):
    if map_data[i] == "-1":
        gap_counter += 1
        index_dict[str(gap_counter)] = i
        continue
    t = int(map_data[i])
    index_dict[df_index[t]] = i

print(index_dict)

#%%
# index_dictをkeyでソート
index_dict = dict(sorted(index_dict.items(), key=lambda x: x[0]))
print(index_dict)

#%%
# index_dictをcsvに保存
with open("/home/sumthen/2023/python_2023/2d_som/result_map/index_dict.csv", "w") as f:
    writer = csv.writer(f)
    for key, value in index_dict.items():
        writer.writerow([key, value])


 # %%
