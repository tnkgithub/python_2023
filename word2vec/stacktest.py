#%%
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()

#%%
df = pd.read_csv('/home/sumthen/2023/python_2023/word2vec/calc_result/jawiki_sum_removed_number_copy.csv', index_col=0)
# %%
stack = np.stack(df.values)
# %%
print(stack.shape)
df["new"] = df['1'].progress_apply(lambda x: np.stack(x))


# %%
def cos_sim_matrix(matrix):
    """文書間のコサイン類似度を計算し、類似度行列を返す"""
    d = matrix @ matrix.T
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    return d / norm / norm.T
# %%
#                                           バーを出す（extractで正規表現にマッチした部分を抽出して分割(str(x))）
stack_sim = cos_sim_matrix(stack)
sim = cos_sim_matrix(np.stack(df.values))
# %%
print(stack_sim)


# %%
print(sim)
# %%
