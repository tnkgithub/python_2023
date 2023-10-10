#%%
from pyknp import Juman
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
jumanpp = Juman()

#%%
wakati = []

for i in range(len(titles)):
    title = titles[i].rstrip('\n')
    wakati.append("title: " + title)

    # 分かち書き
    result = jumanpp.analysis(title)
    for m in result.mrph_list():
        if m.hinsi == '名詞':
            try:
                model[m.midasi]
                wakati.append(m.midasi)
            except:
                wakati.append(m.midasi + ": error")

    if type(captions[i]) != float:
        wakati.append("caption: " + captions[i])
    else:
        wakati.append("caption: " + "nan")
    wakati.append(" ")

str_ = '\n'.join(wakati)
fw = open('title_wakati_juman.txt', 'w')
fw.write(str_)
fw.close()

# %%