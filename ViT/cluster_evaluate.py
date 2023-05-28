#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

#%%
# データの読み込み
df = pd.read_csv('result_feature/feature_vit.csv', index_col=0)

# 階層的クラスタリング
ward_euc = linkage(df, method='ward', metric='euclidean')
fig = plt.figure(figsize=(20, 10), num=None, dpi=80, facecolor='w', edgecolor='k')
dendrogram(ward_euc, labels=df.index, leaf_font_size=8)

# クラスタ数の範囲
NUM_CLUSTERS = range(2, 100)

#%%
# シルエットスコアの計算
def calcSilhouetteScore(df):
    silhouette_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion='maxclust')
        silhouette_scores.append(silhouette_score(df, fclusters))
    fig = plt.figure(figsize=(15, 10), facecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(NUM_CLUSTERS, silhouette_scores,  'bo-', label='Silhouette score')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    plt.show()

    return silhouette_scores

#%%
# ダビスボルディンスコアの計算
def calcDaviesBouldinScore(df):
    davies_bouldin_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion='maxclust')
        davies_bouldin_scores.append(davies_bouldin_score(df, fclusters))

    fig = plt.figure(figsize=(15, 10), facecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(NUM_CLUSTERS, davies_bouldin_scores,  'gs-', label='Davies-Bouldin score')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Davies-Bouldin score')
    plt.show()

    return davies_bouldin_scores

#%%
# カリンスキーハラバススコアの計算
def calcCalinskiHarabaszScore(df):
    calinski_harabasz_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion='maxclust')
        calinski_harabasz_scores.append(calinski_harabasz_score(df, fclusters))

    fig = plt.figure(figsize=(15, 10), facecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(NUM_CLUSTERS, calinski_harabasz_scores,  'rd-', label='Calinski-Harabasz score')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Calinski-Harabasz score')
    plt.show()

    return calinski_harabasz_scores

#%%
# シルエットスコアの計算
silhouette_scores = calcSilhouetteScore(df)

#%%
# ダビスボルディンスコアの計算
davies_bouldin_scores = calcDaviesBouldinScore(df)

#%%
# カリンスキーハラバススコアの計算
calinski_harabasz_scores = calcCalinskiHarabaszScore(df)