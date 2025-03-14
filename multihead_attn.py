'''
输入emb后的词序列,根据Q,K,V方法计算词与词之间的相关性,为每个词生成信息提取后的emb(与输入词1:1映射)
'''
from torch import nn 
import torch 
from dataset import de_vocab,de_preprocess,train_dataset
from emb import EmbeddingWithPosition
import math 
"""
从自然语言序列，经过计算，到隐藏层的过程。
x:  (batch_size,seq_len_q,emb_size)
词之间的先后关系、内容，找一些规律。
"""
class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,head): #尝试理解向量之间的关系 # 多头注意力
        super().__init__()
        self.emb_size=emb_size
        self.q_k_size=q_k_size
        self.v_size=v_size
        self.head=head

        self.w_q=nn.Linear(emb_size,head*q_k_size) # 多头 #把每个头的输出，首尾连接起来 #产生的8个向量，连起来。#
        self.w_k=nn.Linear(emb_size,head*q_k_size)
        self.w_v=nn.Linear(emb_size,head*v_size)

    def forward(self,x_q,x_k_v,attn_mask):
        """
        Transformer的前向传播函数，用于计算多头注意力机制的输出。
        
        Args:
            x_q (Tensor): 查询输入张量，形状为 (batch_size, seq_len_q, emb_size)。
            x_k_v (Tensor): 键和值输入张量，形状为 (batch_size, seq_len_k, emb_size)。
            attn_mask (Tensor): 注意力掩码张量，形状为 (batch_size, seq_len_q, seq_len_k)，用于屏蔽不需要计算注意力的位置。
        
        Returns:
            Tensor: 多头注意力的输出张量，形状为 (batch_size, seq_len_q, head * v_size)。
        
        """
        # x_q: (batch_size,seq_len_q,emb_size)
        q=self.w_q(x_q) # q: (batch_size,seq_len_q,head*q_k_size)
        k=self.w_k(x_k_v) # k: (batch_size,seq_len_k,head*q_k_size)
        
        # 多头兼容
        q=q.view(q.shape[0],q.shape[1],self.head,self.q_k_size).transpose(1,2) # q: (batch_size,head,seq_len_q,q_k_size) #把‘头’这个维度提到前面去 #
        k=k.view(k.shape[0],k.shape[1],self.head,self.q_k_size).transpose(1,2).transpose(2,3) # k:(batch_size,head,q_k_size,seq_len_k)

        # 注意力矩阵
        attn=torch.matmul(q,k)/math.sqrt(self.q_k_size) # (batch_size,head,seq_len_q,seq_len_k) row是q,col是k
        
        # 注意力分值处理
        # attn_mask: (batch_size,seq_len_q,seq_len_k)
        # print(attn_mask.shape) #torch.Size([1, 15, 15])
        # # (batch_size, 1, seq_len_q, seq_len_k) --expand---> (batch_size, head, seq_len_q, seq_len_k)
        attn_mask=attn_mask.unsqueeze(1).expand(-1,self.head,-1,-1) # attn_mask: (batch_size,head,seq_len_q,seq_len_k) #-1表示维度不变 #
        # print(attn_mask.shape) #torch.Size([1, 8, 15, 15])
        # print(attn_mask)

        # print("##attn_mask.shape :", attn_mask.shape)
        # print(attn_mask)
        attn=attn.masked_fill(attn_mask,-1e9)
        attn=torch.softmax(attn,dim=-1) # scores: (batch_size,head,seq_len_q,seq_len_k) #每一行的和为1 #

        # 注意力与V相乘
        v=self.w_v(x_k_v) # v: (batch_size,seq_len_k,head*v_size)
        v=v.view(v.shape[0],v.shape[1],self.head,self.v_size).transpose(1,2) # v: (batch_size,head,seq_len_k,v_size)
        # print(attn.shape) #torch.Size([1, 8, 15, 15])
        # print(attn)
        """[0, head_i, 15, 15]
         [[0.0603, 0.0362, 0.0758,  ..., 0.1247, 0.1180, 0.1080],
          [0.0601, 0.0449, 0.0814,  ..., 0.0472, 0.0410, 0.1834],
          [0.0520, 0.0499, 0.0431,  ..., 0.1585, 0.1277, 0.0859],
          ...,
          [0.0369, 0.0361, 0.0827,  ..., 0.1575, 0.1147, 0.0901],
          [0.0390, 0.0437, 0.0591,  ..., 0.1712, 0.0508, 0.1856],
          [0.0986, 0.0439, 0.0427,  ..., 0.0968, 0.1106, 0.1601]],

        """ 
        # print(v.shape) #torch.Size([1, 8, 15, 512])
        # print(v)
        """[0, head_i, 15, 512]
         [[-0.6026,  0.9561, -1.1390,  ..., -0.3230, -0.5437, -0.3645],
          [ 0.5408,  0.0995, -1.1061,  ..., -0.3467,  0.7035,  0.8775],
          [-0.2972,  0.9936,  0.0683,  ..., -1.2730, -0.0575, -0.5165],
          ...,
          [ 0.5023,  0.2930, -1.6402,  ..., -0.6515, -0.5593,  1.3963],
          [ 1.0337,  0.8440, -1.0133,  ..., -0.3267, -0.3728, -0.2281],
          [ 0.7331,  0.8567, -0.7536,  ..., -0.0116, -0.2361,  0.3307]],
        """
        z=torch.matmul(attn,v) # z: (batch_size,head,seq_len_q,v_size) #v和注意力分数相乘 #
        # print(z.shape) #torch.Size([1, 8, 15, 512])
        # print(z)
        """[0, head-_i, 15, 512]
         [[-1.9871e-02, -6.4012e-02,  1.5275e-01,  ..., -3.6938e-01,
            9.5505e-01, -4.2759e-02],
          [-1.0661e-01, -7.7739e-02,  4.0522e-02,  ..., -2.6471e-01,
            9.8480e-01, -8.8847e-02],
          [-1.1032e-01, -1.9960e-01,  1.2765e-01,  ..., -2.3088e-01,
            9.1152e-01, -1.1021e-01],
          ...,
          [ 1.6569e-02, -1.7410e-01,  1.1236e-01,  ..., -1.1206e-01,
            8.9661e-01, -1.5413e-02],
          [ 1.0616e-01, -2.4999e-01,  2.5896e-01,  ..., -1.6394e-01,
            7.6752e-01,  5.2572e-04],
          [-1.1208e-01, -1.6807e-01,  5.4095e-02,  ..., -1.4015e-01,
            1.0600e+00, -1.1799e-01]],

        """
        z=z.transpose(1,2) # z: (batch_size,seq_len_q,head,v_size)
        return z.reshape(z.shape[0],z.shape[1],-1) # z: (batch_size,seq_len_q,head*v_size) #每个词的多头词向量，首尾相连在一起 #多头的输出结果拼起来 #

if __name__=='__main__':
    # 准备1个batch
    emb=EmbeddingWithPosition(len(de_vocab),128) #词表长度 #
    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    # print('de_tokens:', de_tokens)
    # de_tokens: ['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>']
    # print('de_ids:', de_ids)
    # de_ids: [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    # print(de_ids_tensor.shape) #torch.Size([15])
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    # print('emb_result:', emb_result.shape) #emb_result: torch.Size([1, 15, 128])

    # 多头注意力
    multihead=MultiHeadAttention(emb_size=128,q_k_size=256,v_size=512,head=8)
    attn_mask=torch.zeros((1,de_ids_tensor.shape[0],de_ids_tensor.shape[0])) # batch中每个样本对应1个注意力矩阵 #不遮盖 #
    multihead_result=multihead(x_q=emb_result,x_k_v=emb_result,attn_mask=attn_mask)
    # print('multihead_result:', multihead_result.shape) #multihead_result: torch.Size([1, 15, 4096]) #做语义分析，观察这15个词之间的关系。 #
    # 4096 = 8 * 512 
    # 8头 * 512维 #