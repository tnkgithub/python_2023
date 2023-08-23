#%%
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import os
import csv

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
page_ulrs = []
for i in range(1, last_page + 1):
    url = "http://archives.c.fun.ac.jp/fronts/index/poster/page:{}".format(i)
    page_ulrs.append(url)


#%%
# 各資料のURLを取得
material_urls = []
for url in page_ulrs:
    r = requests.get(url)
    time.sleep(1)

    # ページの情報を取得
    soup = BeautifulSoup(r.text, "html.parser")

    # ページ内の"caption"クラスを取得
    caption = soup.find_all(class_="caption")

    # キャプションからhrefを取得
    for cap in caption:
        a = cap.find("a")
        href = a.attrs["href"]
        material_urls.append(href)

#%%
for i in range(len(material_urls)):
    url = "http://archives.c.fun.ac.jp" + material_urls[i]

    r = requests.get(url)
    time.sleep(1)

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find_all("td")

    metadata = []
    for j in range(len(table)):
        if j % 2 == 1:
            metadata.append(table[j].text)


    for j in range(len(metadata)):
        if metadata[j] == "\xa0":
            metadata[j] = None


    # メタデータの保存
    if i == 0:
        df = pd.DataFrame([metadata])
    else:
        tmp_df = pd.DataFrame([metadata])
        df = pd.concat([df, tmp_df], axis=0)

os.makedirs("./result_metadata", exist_ok=True)
df.to_csv("./result_metadata/metadata_poster.csv", index=False)
# %%
