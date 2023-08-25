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
df = pd.read_csv("../scraping/result_metadata/metadata_poster.csv", index_col=0)
caption = df["1"].to_list()

# %%
nan_label_list = []
result_feature = []
error_label_list = []
# captionから特徴量を抽出
for i, cap in enumerate(caption):
    if type(cap) == float:
        nan_label_list.append(i)
        continue
    input_ids = tokenizer.encode(cap, add_special_tokens=True)
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
    hidden_states_np = [i.numpy() for i in hidden_states]
    concatenated_hidden_np = np.concatenate(hidden_states_np, axis=-1)
    result_feature.append(concatenated_hidden_np)

print(len(result_feature))

# %%
print(len(nan_label_list))
print(len(error_label_list))

# %%
