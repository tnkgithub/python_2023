# %%
import torch
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel, BertConfig

# %%
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)
model_bert = BertModel.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

# %%
# 入力テキストをトークン化
input_text = "今日はいい天気ですね。"
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
tokens_tensor = torch.tensor(input_ids)

# 最終層の出力を得る
with torch.no_grad():
    outputs = model_bert(tokens_tensor.unsqueeze(0))

# 最終4層の隠れ層を取得して結合
hidden_states = outputs.hidden_states

hidden_states = outputs.hidden_states[-4:]
concatenated_hidden = torch.cat(hidden_states, dim=-1)

print(concatenated_hidden.shape)  # 結合後のNumpy配列の形状を表示

# %%
