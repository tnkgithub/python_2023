#%%
import pandas as pd
import numpy as np

#%%
df = pd.read_csv('/content/drive/MyDrive/extract_title_neologd_sudachi_ginza.csv', index_col=0)
titles = df.index.to_list()

#%% 日本語Wikipediaエンティティベクトル
def search(sim_ary, n):
    doc = df.sample(1)
    print(doc)
    doc_idx = titles.index(doc.index[0])
    sim_index = sim_ary[doc_idx].argsort()[::-1]
    rec_df = df.iloc[sim_index][:n]
    rec_df['similarity'] = np.sort(sim_ary[doc_idx])[::-1][:n]
    return rec_df[['similarity']]

sim_ary = np.load('/content/drive/MyDrive/jawiki/jawiki_sum_sim_matrix.npy')
df2 = search(sim_ary, 15)

df2

#%%
def search(sim_ary, n):
    doc = df.sample(1)
    print(doc)
    doc_idx = titles.index(doc.index[0])
    sim_index = sim_ary[doc_idx].argsort()[::-1]
    rec_df = df.iloc[sim_index][:n]
    rec_df['similarity'] = np.sort(sim_ary[doc_idx])[::-1][:n]
    return rec_df[['similarity']]

sim_ary = np.load('/content/drive/MyDrive/fasttext_sum_sim_matrix.npy')
df2 = search(sim_ary, 15)

df2