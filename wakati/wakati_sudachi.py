#%%
from sudachipy import tokenizer
from sudachipy import dictionary
import pandas as pd
import gensim

#%%
# データの読み込み
df = pd.read_csv("/home/b1019035/2023/python_2023/wakati/wakati_2026/wakati_title_no_error.csv", index_col=0)
index = df.index.to_list()
titles = df['title'].to_list()

#%%
print(df['wakati_title'])
# wakati_titleのリストを取得
for i, wakati_title in enumerate(df['wakati_title']):
    wakati_title = wakati_title.replace("[", "")
    wakati_title = wakati_title.replace("]", "")
    wakati_title = wakati_title.replace("'", "")
    wakati_title = wakati_title.split(", ")
    df.at[index[i], 'wakati_title'] = wakati_title

#%%
print(df['wakati_title'])

wakati_titles = df['wakati_title'].to_list()
print(wakati_titles)
#%%
# モデルの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format('./jawiki.all_vectors.300d.txt', binary=False)

# %%
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.A
# mode = tokenizer.Tokenizer.SplitMode.B
# mode = tokenizer.Tokenizer.SplitMode.C

#%%
df_wakati = pd.DataFrame(index=index, columns=['title', 'wakati_title'])

for i, wakati_title in enumerate(wakati_titles):
    df_wakati.at[index[i], 'title'] = titles[i]
    wakati_result = []

    for m in wakati_title:
        # mの型を確認
        for w in tokenizer_obj.tokenize(m, mode):
            if w.part_of_speech()[0] == '名詞':
                try:
                    model[w.surface()]
                    wakati_result.append(w.surface())
                except:
                    continue
    # wakati_titleでdf_wakatiを更新
    df_wakati.at[index[i], 'wakati_title'] = wakati_result

#%%
print(df_wakati)

#%%
df_wakati.to_csv('/home/b1019035/2023/python_2023/wakati/wakati_2026/wakati_title_sudachi_A.csv')
#%%
# wakati = [] # 分かち書きしたタイトル+その他を格納するリスト

# for i, title in enumerate(titles):
#     title = title.rstrip('\n')
#     wakati.append("title: " + title)

#     # 分かち書き
#     for m in tokenizer_obj.tokenize(title, mode):
#         if m.part_of_speech()[0] == '名詞':
#             try:
#                 model[m.surface()]
#                 wakati.append(m.surface())
#             except:
#                 wakati.append(m.surface() + ": error")

#     if type(captions[i]) != float:
#         wakati.append("caption: " + captions[i])
#     else:
#         wakati.append("caption: " + "nan")
#     wakati.append(" ")

# str_ = '\n'.join(wakati)
# # fw = open('title_wakati_sudachi_A.txt', 'w')
# # fw = open('title_wakati_sudachi_B.txt', 'w')
# fw = open('title_wakati_sudachi_C.txt', 'w')
# fw.write(str_)
# fw.close()
# %%
