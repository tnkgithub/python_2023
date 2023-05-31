# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# %%
# データの読み込み
df = pd.read_csv("result_feature/feature_vit.csv", index_col=0)

# 階層的クラスタリング
ward_euc = linkage(df, method="ward", metric="euclidean")
fig = plt.figure(figsize=(20, 10), num=None, dpi=80, facecolor="w", edgecolor="k")
dendrogram(ward_euc, labels=df.index, leaf_font_size=8)

# クラスタ数の範囲
NUM_CLUSTERS = range(2, 100)


# %%
# シルエットスコアを計算する関数
def calcSilhouetteScore(df):
    silhouette_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion="maxclust")
        silhouette_scores.append(silhouette_score(df, fclusters))
    fig = plt.figure(figsize=(15, 10), facecolor="w")
    ax = fig.add_subplot(111)
    ax.plot(NUM_CLUSTERS, silhouette_scores, "bo-", label="Silhouette score")
    ax.set_xlabel("Number of clusters", fontsize=16)
    ax.set_ylabel("Silhouette score", fontsize=16)
    plt.show()

    return silhouette_scores


# %%
# ダビスボルディンスコアを計算する関数
def calcDaviesBouldinScore(df):
    davies_bouldin_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion="maxclust")
        davies_bouldin_scores.append(davies_bouldin_score(df, fclusters))

    fig = plt.figure(figsize=(15, 10), facecolor="w")
    ax = fig.add_subplot(111)
    ax.plot(NUM_CLUSTERS, davies_bouldin_scores, "gs-", label="Davies-Bouldin score")
    ax.set_xlabel("Number of clusters", fontsize=16)
    ax.set_ylabel("Davies-Bouldin score", fontsize=16)
    plt.show()

    return davies_bouldin_scores


# %%
# カリンスキーハラバススコアを計算する関数
def calcCalinskiHarabaszScore(df):
    calinski_harabasz_scores = []
    for i in NUM_CLUSTERS:
        fclusters = fcluster(ward_euc, i, criterion="maxclust")
        calinski_harabasz_scores.append(calinski_harabasz_score(df, fclusters))

    fig = plt.figure(figsize=(15, 10), facecolor="w")
    ax = fig.add_subplot(111)
    ax.plot(
        NUM_CLUSTERS, calinski_harabasz_scores, "rd-", label="Calinski-Harabasz score"
    )
    ax.set_xlabel("Number of clusters", fontsize=16)
    ax.set_ylabel("Calinski-Harabasz score", fontsize=16)
    plt.show()

    return calinski_harabasz_scores


# %%
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

model = KMeans()

visualizer = KElbowVisualizer(model, k=(2, 100), metric="silhouette", timings=False)
visualizer.fit(df)
visualizer.show()

# %%
visualizer = KElbowVisualizer(
    model, k=(2, 100), metric="calinski_harabasz", timings=False
)
visualizer.fit(df)
visualizer.show()

# %%
# シルエットスコアの計算
silhouette_scores = calcSilhouetteScore(df)

# %%
# ダビスボルディンスコアの計算
davies_bouldin_scores = calcDaviesBouldinScore(df)

# %%
# カリンスキーハラバススコアの計算
calinski_harabasz_scores = calcCalinskiHarabaszScore(df)
