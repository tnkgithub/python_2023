#%%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist
import pandas as pd
import matplotlib.pyplot as plt


#%%
df = pd.read_csv('./result_feature_pca/vit_mns.csv', index_col=0)
#df = pd.read_csv('./result_feature_pca/vit_stds.csv', index_col=0)

#%%
pdist = pdist(df, metric='cosine')
ward_euc = linkage(pdist, method='ward', metric='cosine')
plt.figure(figsize=(20, 10), num=None, dpi=80, facecolor='w', edgecolor='k')
dendrogram(ward_euc, labels=df.index, leaf_font_size=8)

# %%
silhouette_coefficient = []
calinski_harabasz_index = []
davies_bouldin_index = []

NUM_CLUSTERS_RANGE = range(10,50)  # クラスター数を2～10個の範囲で比較
for num in NUM_CLUSTERS_RANGE:
    labels = fcluster(ward_euc, t=num, criterion='maxclust')

    silhouette_coefficient.append(silhouette_score(df, labels))
    calinski_harabasz_index.append(calinski_harabasz_score(df, labels))
    davies_bouldin_index.append(davies_bouldin_score(df, labels))

#%%
# グラフの描画
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,10), facecolor='w')
fig.subplots_adjust(bottom=0.3,right=0.75)
host = fig.add_subplot(111)


# 縦軸の追加
par1 = host.twinx()
par2 = host.twinx()

# プロット
p0, = host.plot(NUM_CLUSTERS_RANGE, silhouette_coefficient, 'bo-', label='Silhouette Coefficient')
p1, = par1.plot(NUM_CLUSTERS_RANGE, calinski_harabasz_index, 'rd-', label='Calinski Harabasz Index')
p2, = par2.plot(NUM_CLUSTERS_RANGE, davies_bouldin_index, 'gs-', label='Davies Bouldin Index')

# 軸ラベル
host.set_xlabel('Number of Clusters')
host.set_ylabel('Silhouette Coefficient')
par1.set_ylabel('Calinski Harabasz Index')
par2.set_ylabel('Davies Bouldin Index')

# 軸の位置の調整
par2.spines['right'].set_position(('axes', 1.15))

# 凡例
lines = [p0, p1, p2]
host.legend(lines,
            [l.get_label() for l in lines],
            fontsize=15,
            bbox_to_anchor=(0.7, -0.1),
            loc='upper left')

fig.show()
##fig.savefig('./result_cluster_vit/mns_ward_cos_evaluate.png')
# %%
# 適切なクラスター数を決定
# zが最大のクラスター数を取得
max_silhouette_coefficient = max(silhouette_coefficient)
optimal_num_clusters = silhouette_coefficient.index(max_silhouette_coefficient) + 2
print('Optimal number of clusters: {}'.format(optimal_num_clusters))

# カリンスキー・ハラバス指数が最大のクラスター数を取得
max_calinski_harabasz_index = max(calinski_harabasz_index)
optimal_num_clusters = calinski_harabasz_index.index(max_calinski_harabasz_index) + 2
print('Optimal number of clusters: {}'.format(optimal_num_clusters))

# ダヴィス・ボルディン指数が最小のクラスター数を取得
min_davies_bouldin_index = min(davies_bouldin_index)
optimal_num_clusters = davies_bouldin_index.index(min_davies_bouldin_index) + 2
print('Optimal number of clusters: {}'.format(optimal_num_clusters))

# %%
# シルエット数 + カリンスキー・ハラバス指数 - ダヴィス・ボルディン指数 が最大のクラスター数を取得
max_score = 0
optimal_num_clusters = 0
for i in range(len(NUM_CLUSTERS_RANGE)):
    score = silhouette_coefficient[i] + calinski_harabasz_index[i] - davies_bouldin_index[i]
    if score > max_score:
        max_score = score
        optimal_num_clusters = i + 10
print('Optimal number of clusters: {}'.format(optimal_num_clusters))

# %%