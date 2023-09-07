import torch

# バッチサイズとシーケンス長を定義
batch_size = 2
length = 5

# ダミーの入力テンソルとマスクを生成
input = torch.tensor([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10]])

mask = torch.tensor([[True, False, True, False, True],
                     [False, True, False, True, False]])

# masked_selectを適用して要素を選択
selected_elements = torch.masked_select(input, mask)

print(selected_elements)
breakpoint()