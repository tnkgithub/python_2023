#%%
import os
import cv2

#%%
img_dir_path =   '/home/sumthen/dev/theme-app/web/public/posters/normalPoster'
img_name = os.listdir(img_dir_path)

#%%
def resize_and_trim(img, width, height):
    h, w = img.shape[:2]
    magnification = height / h
    reWidth = int(magnification * w)
    size = (reWidth, height)

    if reWidth < 100:
        magnification = width / w
        reHeight = int(magnification * h)
        size = (width, reHeight)

    img_resize = cv2.resize(img, size)

    h, w = img_resize.shape[:2]

    top = int((h / 2) - (height / 2))
    bottom = top+height
    left = int((w / 2) - (width / 2))
    right = left+width

    return img_resize[top:bottom, left:right]


#%%
trimmed_img_dir_path = '/home/sumthen/dev/theme-app/web/public/posters/trimmedPoster'
os.makedirs(trimmed_img_dir_path, exist_ok=True)

for i in img_name:
    img = cv2.imread(img_dir_path + '/' + i)
    img = resize_and_trim(img, 100, 162)
    cv2.imwrite(trimmed_img_dir_path + '/' + i, img)

# %%
print(len(img_name))
# %%
