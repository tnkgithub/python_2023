#%%
import csv

#%%
#'/home/sumthen/2023/python_2023/scraping/numbers.csv'を読み込む
# 全行に ' Float'を追加して上書き保存する。型変換はしない. カンマ区切りにしない
with open('numbers.csv', 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

with open('numbers_parsefloat.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow([f'{cell}: parseFloat(row['+''+cell+']),' for cell in row])
        # writer.writerow([cell+': parseFloat(row['+cell+']),' for cell in row])
# %%
