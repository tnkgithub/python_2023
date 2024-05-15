#%%
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import os

#%%
main_url = "https://archives.c.fun.ac.jp/posters"
page_template = "?page={}"
last_page = 169

images_name_list = []
explan_page_list = []

for i in range(1, last_page + 1):
    page_url = main_url + page_template.format(i)

    r = requests.get(page_url)
    time.sleep(1)

    soup = BeautifulSoup(r.text, "html.parser")
    get_images = soup.find_all("img")

    for n in range(len(get_images)):
        img_num = get_images[n].attrs["src"].split("/")[4] + "00" + ".jpg"
        page_link = "/" + get_images[n].attrs["src"].split("/")[4] + "/" + get_images[n].attrs["src"].split("/")[5]

        images_name_list.append(img_num)
        explan_page_list.append(main_url + page_link)


# %%
# 2つのリストをデータフレームに変換
df = pd.DataFrame({"col1": images_name_list, "col2": explan_page_list})

# %%
df.to_csv("img_link.csv", index=False)
# %%
