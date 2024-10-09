#%%
import MeCab as mc
import numpy as np
import pandas as pd
import gensim
#import ipadic

#%%
# データの読み込み
df = pd.read_csv("/home/b1019035/2023/python_2023/scraping/metadata_2026.csv", index_col=0, header=None)
index = df.index.to_list()
titles = df[1].to_list()

#%%
# モデルの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format('./jawiki.all_vectors.300d.txt', binary=False)

# %%
# 分かち書き用のTagger
tagger = mc.Tagger('-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
# ipa辞書を使う場合
#tagger = mc.Tagger(ipadic.MECAB_ARGS)
# uniDicを使う場合
#tagger = mc.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/")

# %%  タイトルの分かち書き（Word2Vecのエエラーも）
# df: index, title, wakati_title[]となるデータフレームを作成
df_wakati = pd.DataFrame(index=index, columns=['title', 'wakati_title'])

for i, title in enumerate(titles):
    # タイトルから改行コードを削除
    title = title.rstrip('\n')
    # print(title)
    df_wakati.at[index[i], 'title'] = title

    # 分かち書き
    node = tagger.parseToNode(title)

    # 名詞のみを抽出し、Word2Vecのモデルに存在するかを確認
    wakati_title = []
    while node:
        print(node.surface)
        if node.feature.split(",")[0] == '名詞':
        # nodeが名詞で、かつWord2Vecのモデルに存在する単語のみを抽出
        # if 36 <= node.posid <= 67:
            # try:
            #     model[node.surface]
            #     wakati_title.append(node.surface)
            # except:
            #     wakati_title.append(node.surface + ": error")
            wakati_title.append(node.surface)
        node = node.next

    df_wakati.at[index[i], 'wakati_title'] = wakati_title

#%%
print(df_wakati)
#%%
# 分かち書きしたタイトルをファイルに書き込む
df_wakati.to_csv('/home/b1019035/2023/python_2023/wakati/wakati_2026/wakati_title_no_error.csv')

#%%
str_ = '\n'.join(title_wakati)
fw = open('title_wakati_ipa.txt', 'w')
fw.write(str_)
fw.close()
# %%
