#%%
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
import os
import pandas as pd

#%%
img_dir_path = "/home/b1019035/dev/theme-app/web/public/posters/normalPoster"
img_list = os.listdir(img_dir_path)

#%%
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)


#%%

def extract_feature(img_path):
    img = Image.open(img_path)
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_token_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_token_embedding

#%%
for_vstack = torch.empty([0, 768])
for i in img_list:
    img_path = os.path.join(img_dir_path, i)
    feature = extract_feature(img_path)
    for_vstack = torch.vstack((for_vstack, feature))

#%%
print(for_vstack.shape)

#%%
os.makedirs('../ViT/result_feature', exist_ok=True)

#%%
df = pd.DataFrame(for_vstack.numpy(), index=img_list)
df.to_csv("../ViT/result_feature/feature_vit_normal_HF.csv")