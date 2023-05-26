#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#%%
df = pd.read_csv('result_feature/feature_vit.csv', index_col=0)
features = df.to_numpy()
features = features.astype(np.float32)

#%%
tsne = TSNE(n_components=2, random_state=0)
reduced_data = tsne.fit_transform(features)

#%%
Figure = plt.figure(figsize=(10, 10))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='o', s=100, alpha=0.5)
plt.show()

#%%