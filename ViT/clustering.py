# %%
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pandas as pd
import os
import datetime

# %%
#df = pd.read_csv("./result_feature/feature_vit.csv", index_col=0)
df = pd.read_csv("/home/b1019035/2023/python_2023/ViT/result_feature/feature_vit_normal_HF.csv", index_col=0)
# new_index = []
# for i in range(len(df)):
#     new_name = df.index[i].split("-")[0]
#     new_name = new_name[:8]
#     new_index.append(new_name)
#     print(new_name)
img_dir_path = "/home/b1019035/dev/theme-app/web/public/posters/normalPoster"

# %%
pdist = pdist(df, metric="cosine")
ward_cos = linkage(pdist, method="ward", metric="cosine")
plt.figure(figsize=(20, 10), num=None, dpi=80, facecolor="w", edgecolor="k")
# dendrogram(ward_cos, labels=new_index, leaf_font_size=8)
dendrogram(ward_cos, labels=df.index, leaf_font_size=8)

# %%
bucket = [0] * 2000  # クラスタ数、サイズが分からないので40としておく
cluster_num = 0
clusters_size = [0]

#fclusters = fcluster(ward_cos, 1.0, criterion="distance")
fclusters = fcluster(ward_cos, 101, criterion="maxclust")

# クラスターの数と各クラスターのサイズを求める
for i in fclusters:
    # 各クラスターのサイズを一時的に保存
    bucket[i] += 1

for i in bucket:
    if i != 0:
        cluster_num += 1
        clusters_size.append(i)

print(cluster_num)
print(clusters_size)

# %%
cluster_list = [[] for i in range(cluster_num + 1)]
for i in range(len(fclusters)):
    # img_path = os.path.join(img_dir_path, new_index[i])
    img_path = os.path.join(img_dir_path, df.index[i])
    # img_path = img_path + ".jpg"
    print(img_path)
    cluster_list[fclusters[i]].append(img_path)

#%%
os.mkdir('/home/b1019035/2023/python_2023/ViT/result/result_cluster_vit_hf')

# %%
m = 5
n = 18
now = datetime.datetime.now()
for clust_no in range(1, cluster_num):

    imgs = [0] * (len(cluster_list[clust_no]) + 1)
    img_name = (
        '/home/b1019035/2023/python_2023/ViT/result/result_cluster_vit_hf/cluster_'
        + str(clust_no)
        + "_"
        + now.strftime("%Y%m%d_%H%M%S")
        + ".png"
    )



    plt.figure(figsize=(10, 60), facecolor="w")
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, name in zip(range(m * n), cluster_list[clust_no]):
        # 画像を読み込む
        imgs[i] = cv2.imread(name)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        plt.subplot(n, m, i + 1)
        plt.subplots_adjust(hspace=0.0)
        plt.axis("off")
        plt.imshow(imgs[i])

    plt.savefig(img_name)

# %%