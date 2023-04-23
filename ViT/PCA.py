#%%
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os

#%%
df = pd.read_csv('./result_feature/feature_vit.csv', index_col=0)

features = df.to_numpy()

features = features.astype(np.float32)

#%%
pca = PCA(n_components=1978)
pca.fit(features)
features_pca = pca.transform(features)
print(features_pca[0])
print(features_pca.shape)
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

#%%
#os.mkdir('./result_feature_pca')
#%%
pca_df = pd.DataFrame(features_pca, index=df.index)
pca_df.to_csv('./result_feature_pca/vit_pca.csv')

#%%