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
    title = title.text.replace("\n", "").replace(" ", "")
    text = metadata.text.replace("\n", "").replace(" ", "")

    tmp_data = [i, title, text]
    material_metadata.append(tmp_data)

#%%
import csv
with open("/home/sumthen/2023/python_2023/scraping/raw_metadata_2026.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(material_metadata)


# %%
import pandas as pd
colmns = ["資料番号", "タイトル", "内容説明", "西暦"]

table = pd.Series(split_metadata)
print(table)

#%%
df = pd.DataFrame(split_metadata, columns=colmns)
#%%
df.to_csv("/home/sumthen/2023/python_2023/scraping/metadata_2026.csv", index=False, header=False)
# %%
df_1 = df[0].str.split(",", expand=True)
# %%
df_1.to_csv("../scraping/metadata.csv", index=False, header=False)
# %%
