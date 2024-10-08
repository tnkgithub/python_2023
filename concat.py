#%%
import pandas as pd

#%%
df_metadata = pd.read_csv("/home/b1019035/2023/python_2023/scraping/test_metadata_2026.csv", header=None, index_col=0)
print(df_metadata.shape)

#%%
df_som_coordinat = pd.read_csv("/home/b1019035/2023/python_2023/2d_som/result_map/map_index_number.csv", header=None, index_col=0)
print(df_som_coordinat)

#%%
df_reprerentation = pd.read_csv("/home/b1019035/2023/python_2023/2d_som/result_map/representative_number.csv", header=None, index_col=0)
print(df_reprerentation[1].astype(int))
df_reprerentation=df_reprerentation[1].astype(int)
# %%
# df_concat = pd.concat([df_metadata, df_som_coordinat], axis=1)
# print(df_concat.shape)
# df_concat.to_csv("/home/b1019035/2023/python_2023/metadata_map.csv", header=False)

# %%
df_concat = pd.read_csv("/home/b1019035/2023/python_2023/metadata_map.csv", header=None, index_col=0)
df_concat = pd.concat([df_concat, df_reprerentation], axis=1)
print(df_concat.dtypes)
df_concat[4]
df_concat.to_csv("/home/b1019035/2023/python_2023/metadata_map_rep.csv", header=False)
# %%
