#%%
import pandas as pd
import csv

#%%
# somの結果のmapを読み込む
df_map = pd.read_csv('/home/sumthen/2023/python_2023/2d_som/result_map/result_20240904_090803.csv', header=None)
df_feaure = pd.read_csv('/home/sumthen/2023/python_2023/ViT/result_feature/feature_vit_normal_HF.csv', index_col=0)

# %%
# mapを1次元リストに変換
map_list = df_map.values.tolist()
map_list = [j for i in map_list for j in i]
print(map_list)
# %%
# feature_indexをリストに変換
feature_index = df_feaure.index.tolist()
print(feature_index)
# %%
# map_listのindexをfeature_indexに変換
map_list = [feature_index[i] for i in map_list]
print(map_list)
# %%
# map_listをcsvに書き込む
with open('/home/sumthen/dev/theme-app/web/public/tmp/resultSom/som_result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(map_list)

# %%
