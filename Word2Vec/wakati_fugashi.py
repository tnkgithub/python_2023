#%%
from fugashi import Tagger
import pandas as pd
import gensim

#%%
df = pd.read_csv("../scraping/result_metadata/metadata_poster.csv", index_col=0)
captions = df["1"].to_list()
titles = df.index.to_list()

#%%
model = gensim.models.KeyedVectors.load_word2vec_format('./jawiki.all_vectors.300d.txt', binary=False)

# %%about:blank#blocked
tagger = Tagger('-Owakati')

#%%
title_wakati = [] # 分かち書きしたタイトルを格納するリスト

for i, title in enumerate(titles):
    title = title.rstrip('\n')
    title_wakati.append("title: " + title)

    # 分かち書き
    node = tagger(title)

    for node in node:
        if node.feature.pos1 == '名詞':
            try:
                model[node.surface]
                title_wakati.append(node.surface)
            except:
                title_wakati.append(node.surface + ": error")

    if type(captions[i]) != float:
        title_wakati.append("caption: " + captions[i])
    else:
        title_wakati.append("caption: " + "nan")
    title_wakati.append(" ")

str_ = '\n'.join(title_wakati)
fw = open('title_wakati_fugashi.txt', 'w')
fw.write(str_)
fw.close()


# %%
