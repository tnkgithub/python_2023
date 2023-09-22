#%%
from bs4 import BeautifulSoup
import requests
import time
import os

#%%
# 取得したいサイトのURL
main_url = "http://archives.c.fun.ac.jp/fronts/index/poster"

# リンクを呼び出す
r = requests.get(main_url)
time.sleep(1)

# サイト内の情報をすべて取得
soup = BeautifulSoup(r.text, "html.parser")
# ページ遷移クラスの取得
page_nation = soup.find(class_="pagination")
# ページ遷移クラス内の"a"タグを取得
page_num = page_nation.find_all("a")

pages = []
# "a"タグ内のページ番号を取得
for i in page_num:
    pages.append(i.text)


#%%
# 最後のページ番号
last_page = int(pages[-3])

# 全ページのURLをリストに加える
page_urls = []
for i in range(1, last_page + 1):
    url = "http://archives.c.fun.ac.jp/fronts/index/poster/page:{}".format(i)
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

os.makedirs("./images", exist_ok=True)

#%%
for image in images_link_list:
    r = requests.get(image)
    time.sleep(1)
    image_file =  open('./images/' + image.split('/')[-3] + image.split('/')[-1], mode='wb')
    image_file.write(r.content)
    image_file.close()

files = os.listdir("./images")
print("画像の枚数：", len(files))

#%%