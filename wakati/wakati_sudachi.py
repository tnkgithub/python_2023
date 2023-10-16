#%%
from sudachipy import tokenizer
from sudachipy import dictionary
import pandas as pd
import gensim

#%%
# データの読み込み
df = pd.read_csv("../scraping/result_metadata/metadata_poster.csv", index_col=0)
captions = df["1"].to_list()
titles = df.index.to_list()

#%%
# モデルの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format('./jawiki.all_vectors.300d.txt', binary=False)

# %%
tokenizer_obj = dictionary.Dictionary().create()
# mode = tokenizer.Tokenizer.SplitMode.A
# mode = tokenizer.Tokenizer.SplitMode.B
mode = tokenizer.Tokenizer.SplitMode.C

#%%
wakati = [] # 分かち書きしたタイトル+その他を格納するリスト

for i, title in enumerate(titles):
    title = title.rstrip('\n')
    wakati.append("title: " + title)

    # 分かち書き
    for m in tokenizer_obj.tokenize(title, mode):
        if m.part_of_speech()[0] == '名詞':
            try:
                model[m.surface()]
                wakati.append(m.surface())
            except:
                wakati.append(m.surface() + ": error")

    if type(captions[i]) != float:
        wakati.append("caption: " + captions[i])
    else:
        wakati.append("caption: " + "nan")
    wakati.append(" ")

str_ = '\n'.join(wakati)
# fw = open('title_wakati_sudachi_A.txt', 'w')
# fw = open('title_wakati_sudachi_B.txt', 'w')
fw = open('title_wakati_sudachi_C.txt', 'w')
fw.write(str_)
fw.close()
# %%
