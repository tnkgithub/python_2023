# %%
import torch
import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel, BertConfig

# %%
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model_bert = BertModel.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking", output_hidden_states=True
)

# %%
df = pd.read_csv("../scraping/result_metadata/metadatabout:blank#blockeda_poster.csv", index_col=0)
caption = df["1"].to_list()

# %%
nan_label_list = []
result_feature = np.empty((0, 768 * 4 * 4))
error_label_list = []

ids_list = []
# captionから特徴量を抽出
for i, cap in enumerate(caption):
    if type(cap) == float:
        nan_label_list.append(i)
        continue

    # input_ids = tokenizer.encode(cap, add_special_tokens=True)
    tokens = tokenizer.tokenize(cap)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(len(input_ids))

    # ids = tokenizer.convert_ids_to_tokens(input_ids)
    # ids_list.append(ids)

    # df_ids = pd.DataFrame(ids_list)
    # df_ids.to_csv("ids.csv")
    #print(tokenizer.convert_ids_to_tokens(input_ids))

    tokens_tensor = torch.tensor(input_ids)

    # 隠れ層の出力を得る
    with torch.no_grad():
        try:
            outputs = model_bert(tokens_tensor.unsqueeze(0))
        except:
            error_label_list.append(i)
            continue

    # 最終4層の隠れ層を取得して結合
    hidden_states = outputs.hidden_states[-4:]

    print(hidden_states[0].shape)
    print(hidden_states)
    hidden_states_np = [i.numpy() for i in hidden_states]
    print(hidden_states_np[0].shape)
    print(hidden_states_np)
    concatenated_hidden_np = np.concatenate(hidden_states_np, axis=-1)
    print(concatenated_hidden_np.shape)
    print(concatenated_hidden_np)
    break
    #result_feature = np.append(result_feature, concatenated_hidden_np, axis=0)

#print(result_feature.shape)

#%%
print(result_feature)
# %%
print(len(nan_label_list))
print(len(error_label_list))

# %%
print(nan_label_list)

# %%
print(error_label_list)
# %%
