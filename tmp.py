import torch

# 创建一个4x4的attn矩阵
attn = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]])

attn_mask = torch.tensor([[0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])

attn=attn.masked_fill(attn_mask,-1e9)
print("##before:", attn)
attn=torch.softmax(attn,dim=-1) 
print("##after:", attn)