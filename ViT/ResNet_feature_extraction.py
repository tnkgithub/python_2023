#%%
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
from sklearn.decomposition import PCA
import pandas as pd

# %%

model = InceptionResNetV2(weights='imagenet', include_top=False)
img_dir_path = '/home/b1019035/python/gra_study/imagesSub'

img_list = os.listdir(img_dir_path)

#%%
for_vstack = np.empty([0, 98304])
for i in img_list:
    img_path = os.path.join(img_dir_path, i)
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    flatten_features = np.ravel(features)
    for_vstack = np.vstack((for_vstack, flatten_features))

#%%
#os.mkdir('./result_feature')

#%%
df = pd.DataFrame(for_vstack, index=img_list)
df.to_csv('./result_feature/feature_resnet.csv')
