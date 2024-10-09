#%%
from bs4 import BeautifulSoup
import requests
import time
import os

#%%
# 取得したいサイトのURL
main_url = "https://archives.c.fun.ac.jp/posters"

r = requests.get(main_url)
time.sleep(0.5)
soup = BeautifulSoup(r.text, "html.parser")
# エリアラベルの取得
page_nation = soup.find_all('nav', attrs={'aria-label': 'pager'})

#%%
# ページ数の取得
pages = []
for i in page_nation:
    a = i.find_all('a')
    for j in a:
        pages.append(j.text)

# %%
# 最後のページ番号
last_page = int(pages[-2])

# 全ページのURLをリストに加える
page_urls = []
for i in range(1, last_page + 1):
    url = os.path.join(main_url, "?page={}".format(i))
    page_urls.append(url)


#%%
# 画像のURLを取得
images_link_list = []

for page in page_urls:
    r = requests.get(page)
    time.sleep(1)
    soup = BeautifulSoup(r.text, "html.parser")
    get_images = soup.find_all("img")

    for n in range(len(get_images)):
        images_link_list.append(get_images[n].attrs["src"])



# %%

os.makedirs("/home/sumthen/2023/python_2023/scraping/images_2026", exist_ok=True)

#%%
for image in images_link_list:
    r = requests.get(image)
    time.sleep(1)
    image_file =  open('./images/' + image.split('/')[-3] + ".jpg", mode='wb')
    image_file.write(r.content)
    image_file.close()

files = os.listdir("../scraping/images")
print("画像の枚数：", len(files))

#%%

img_dir_path = "/home/sumthen/2023/python_2023/scraping/images_2026"
img_list = os.listdir(img_dir_path)
print(len(img_list))
# %%
print(img_list[1])
# %%
