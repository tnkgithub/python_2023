# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# %%
# 画像ディレクトリのパス
imgs_dir_path = "./image_postcard"


# %%  画像情報を取得
imgs_path = []
img_list = os.listdir(imgs_dir_path)
for i in img_list:
    imgs_path.append(imgs_dir_path + i)


# %%
# 画像ファイルが破損していないかを確認
from pathlib import Path
import imghdr

image_extensions = [".png", ".jpg"]
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(imgs_dir_path).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")


# %%
