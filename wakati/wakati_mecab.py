#%%
import MeCab as mc
import numpy as np
import pandas as pd
import gensim
import ipadic

#%%
# データの読み込み
df = pd.read_csv("../scraping/result_metadata/metadata_poster.csv", index_col=0)
captions = df["1"].to_list()
titles = df.index.to_list()

#%%
# モデルの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format('./jawiki.all_vectors.300d.txt', binary=False)

# %%
# 分かち書き用のTagger
tagger = mc.Tagger('-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
# ipa辞書を使う場合
#tagger = mc.Tagger(ipadic.MECAB_ARGS)
# uniDicを使う場合
#tagger = mc.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/")

# %%  タイトルの分かち書き（Word2Vecのエエラーも）
title_wakati = [] # 分かち書きしたタイトルを格納するリスト

for i, title in enumerate(titles):
    # タイトルから改行コードを削除
    title = title.rstrip('\n')
    title_wakati.append("title: " + title)

    # 分かち書き
    #tagger.parseToNode("")
    node = tagger.parseToNode(title)
    print(node.surface)

    # 名詞のみを抽出し、Word2Vecのモデルに存在するかを確認
    while node:
        if node.feature.split(",")[0] == '名詞':
        # nodeが名詞で、かつWord2Vecのモデルに存在する単語のみを抽出
        #if 36 <= node.posid <= 67:
            try:
                model[node.surface]
                title_wakati.append(node.surface)
            except:
                title_wakati.append(node.surface + ": error")
        node = node.next

    if type(captions[i]) != float:
        title_wakati.append("caption: " + captions[i])
    else:
        title_wakati.append("caption: " + "nan")
    title_wakati.append(" ")

#%%


#%%
str_ = '\n'.join(title_wakati)
fw = open('title_wakati_ipa.txt', 'w')
fw.write(str_)
fw.close()
# %%
