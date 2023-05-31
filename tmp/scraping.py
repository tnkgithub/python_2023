# %%
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import os
import csv

# %%
# 取得したいサイトのURL
main_url = "http://archives.c.fun.ac.jp/fronts/index/postcards"

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


# %%
# 最後のページ番号
last_page = int(pages[-3])
"""全ページのURLをリストに加える"""
urls = []
for i in range(1, last_page + 1):
    url = "http://archives.c.fun.ac.jp/fronts/index/postcards/page:{}".format(i)
    urls.append(url)


# %%

# titles = []
images = []
"""画像とタイトルを取得"""
for url in urls:
    # ページのリンクを呼び出す
    r = requests.get(url)
    time.sleep(1)

    # ページの情報を取得
    soup = BeautifulSoup(r.text, "html.parser")

    # # ページ内の"caption"クラスを取得
    # caption = soup.find_all(class_="caption")

    # # キャプションからタイトルを取得
    # for cap in caption:
    #     a = cap.find("a")
    #     title = a.attrs["title"]
    #     titles.append(title)

    # ページ内の"img"タグを取得
    img_tag = soup.find_all("img")

    # 画像をすべて取得
    for i in img_tag:
        image_link = i.attrs["src"]
        images.append(image_link)
        print(image_link)


# %%
img_title = [i.split("/")[5] for i in images for i in images]
print(img_title)
print(len(img_title))


# %%  タイトルをテキスト形式で出力


# %%
# dict = {}
# for i in range(len(replace_list)):
#     split_name = images[i].split("/")
#     name = split_name[5] + split_name[7]
#     title = replace_list[i] + str(i)
#     dict.setdefault(title, name)

# f = open("title_imageName_dict.csv", "w")
# writer = csv.writer(f)
# writer.writerow(["col1", "col2"])
# for i, j in dict.items():
#     writer.writerow([i, j])
# f.close()


# %%  画像を保存
# imagesファイルをディレクトリ下に作成
# os.mkdir("./image_postcard")


"""画像を保存"""
for i in range(len(images)):
    # 画像urlを呼び出す
    r = requests.get(images[i])
    time.sleep(1)

    # imagesファイルを開き、画像名を決定しながら画像を保存
    img_file = open("./image_postcard/" + img_title[i] + ".jpg", mode="wb")
    img_file.write(r.content)
    img_file.close()


# %%
files = os.listdir("./imageTitle1")
print("画像を" + str(len(files)) + "枚保存しました。")

# %%
