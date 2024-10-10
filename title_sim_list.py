#%%
import pandas as pd

#%%
df = pd.read_csv('/home/sumthen/dev/theme-app/web/public/tmp/data/jawiki_sum_removed_number_sim_matrix_2026_r2.csv', index_col=0)
index = df.index.to_list()
sim_list = df.values
# %%
print(index)
print(type(sim_list[0][0]))

# %%
header_list = [f'sim_{i}' for i in range(1, 201)]
header_list.insert(0, 'id')
# %%
new_df = pd.DataFrame(index=index, columns=header_list)
# %%
print(new_df)

# %%
for i in index:
