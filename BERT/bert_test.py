#%%
import torch
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel, BertConfig

#%%
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

#%%
text = '今日はいい天気ですね。'
input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

with torch.no_grad():
    outputs = model_bert(input_ids)

hidden_states = outputs.last_hidden_state
concatenated_hidden = torch.cat(hidden_states[-4:], dim=-1)

#%%
# 入力テキストをトークン化
input_text = '今日はいい天気ですね。'
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
tokens_tensor = torch.tensor(input_ids)

# 隠れ層の出力を得る
with torch.no_grad():
    outputs = model_bert(tokens_tensor.unsqueeze(0))

# 最終4層の隠れ層を取得して結合
hidden_states = outputs.hidden_states

print(hidden_states[-1].shape)  # 最終層の形状を表示
#%%
hidden_states_np = [layer.numpy() for layer in hidden_states[-4:]]
concatenated_hidden_np = np.concatenate(hidden_states_np, axis=-1)

print(concatenated_hidden_np.shape)  # 結合後のNumpy配列の形状を表示

# %%
