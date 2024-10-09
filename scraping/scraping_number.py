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

# %%
# 資料番号を取得
material_nums = []

for page in page_urls:
    r = requests.get(page)
    time.sleep(0.5)
    soup = BeautifulSoup(r.text, "html.parser")
    get_materials_list = soup.find_all('div', style="-webkit-line-clamp: 1; overflow: hidden;display: -webkit-box;-webkit-box-orient: vertical;")

    for i in get_materials_list:
        num = i.text.strip()
        material_nums.append(num)
    print(page)

# %%
# ファイルの保存
import pandas as pd
df = pd.DataFrame(material_nums)

df.to_csv("../scraping/numbers.csv", index=False, header=False)

# %%
# メタデータの取得
material_metadata = []

for i in material_nums:
    url = os.path.join(main_url, i, "0001")
    r = requests.get(url)
    time.sleep(0.5)
    soup = BeautifulSoup(r.text, "html.parser")
    # タイトルとメタデータの取得
    title = soup.find('div', class_="text-2xl font-medium mb-3")
    metadata = soup.find('div', class_="bg-white mt-10 px-2 md:px-6 lg:px-32 py-6")

    # メタデータの整形
    title = title.text.replace("\n", "").replace("  ", "").replace(",", "、").replace("\r", "、")
    text = ''.join(metadata.text.splitlines())
    text = text.replace(" ", "").replace(",", "、").replace("　", "、")
    # text = metadata.text.replace("\n", "").replace(" ", "").replace(",", "、").replace("\r", "、")

    split_metadata = text.split("内容説明：")[1].split("出版者：")[0]

    tmp_data = i + "," + title + "," + split_metadata
    material_metadata.append(tmp_data)

print(material_metadata)
#%%
import csv
with open("/home/b1019035/2023/python_2023/scraping/_metadata_2026_space.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(material_metadata)


# %%
# メタデータのソート
sorted_metadata = sorted(material_metadata, key=lambda x: x[0])
print(sorted_metadata)
#%%
import pandas as pd
colmns = ["資料番号", "タイトル", "内容説明"]

table = pd.Series(sorted_metadata)
print(table)

#%%
df = pd.DataFrame(table)
df = df[0].str.split(",", expand=True)
print (df.head())
#%%
df.to_csv("/home/b1019035/2023/python_2023/scraping/test_metadata_2026.csv", index=False, header=False)
# %%
