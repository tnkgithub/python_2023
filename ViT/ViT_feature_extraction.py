# %%
from vit_keras import vit
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd

# %%
model = vit.vit_b16(
    image_size=384,
    activation="sigmoid",
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)

img_dir_path = "/home/b1019035/python/gra_study/imagesSub"
img_list = os.listdir(img_dir_path)

# %%
for_vstack = np.empty([0, 768])
for i in img_list:
    img_path = os.path.join(img_dir_path, i)
    img = image.load_img(img_path, target_size=(384, 384))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vit.preprocess_inputs(x)
    features = model.predict(x)
    flatten_features = np.ravel(features)
    for_vstack = np.vstack((for_vstack, flatten_features))

# %%
# os.mkdir('./result_feature')

# %%
df = pd.DataFrame(for_vstack, index=img_list)
df.to_csv("./result_feature/feature_vit.csv")
# %%
