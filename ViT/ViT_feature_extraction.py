# %%
from vit_keras import vit
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
import sys



# %%
model = vit.vit_b16(
    image_size=384,
    activation="sigmoid",
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)

#%%
# df = pd.read_csv("/home/b1019035/dev/theme-app/web/public/posters/normalPoster", index_col=0)
# img_list = df["6"].to_list()

# img_dir_path = "../scraping/images"

img_dir_path = "/home/b1019035/dev/theme-app/web/public/posters/normalPoster"
img_list = os.listdir(img_dir_path)


# %%
for_vstack = np.empty([0, 768])
for i in img_list:
    # i = i + ".jpg"
    # img_path = os.path.join(img_dir_path, i)
    img_path = os.path.join(img_dir_path, i)
    img = image.load_img(img_path, target_size=(384, 384))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vit.preprocess_inputs(x)
    features = model.predict(x)
    flatten_features = np.ravel(features)
    for_vstack = np.vstack((for_vstack, flatten_features))

# %%
os.mkdir('../ViT/result_feature')

# %%
df = pd.DataFrame(for_vstack, index=img_list)
df.to_csv("../ViT/result_feature/feature_vit_normal.csv")
# %%
