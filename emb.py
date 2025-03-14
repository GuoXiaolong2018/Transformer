'''
输入词序列，先做id向量化,再给id附加位置信息
'''
from torch import nn 
import torch 
from dataset import de_vocab,de_preprocess,train_dataset
import math 

class EmbeddingWithPosition(nn.Module):
    def __init__(self,vocab_size,emb_size,dropout=0.1,seq_max_len=5000):
        """
        初始化方法。
        
        Args:
            vocab_size (int): 词汇表的大小。
            emb_size (int): 词嵌入的维度大小。
            dropout (float, optional): Dropout比率，默认为0.1。
            seq_max_len (int, optional): 序列的最大长度，默认为5000。
        
        Returns:
            None
        
        """
        super().__init__()

        # 序列中的每个词转emb向量, 其他形状不变
        self.seq_emb=nn.Embedding(vocab_size,emb_size) #需要被训练的参数 #把每一个词的向量训练好 #

        # 为序列中每个位置准备一个位置向量，也是emb_size宽
        position_idx=torch.arange(0,seq_max_len,dtype=torch.float).unsqueeze(-1) #torch.Size([15, 1])
        # print('position_idx:', position_idx)
        #position_idx: tensor([[0.0000e+00],
        # [1.0000e+00],
        # [2.0000e+00],
        # ...,
        # [4.9970e+03],
        # [4.9980e+03],
        # [4.9990e+03]])
        # print('position_idx:', position_idx.shape) #position_idx: torch.Size([5000, 1])
        position_emb_fill=position_idx*torch.exp(-torch.arange(0,emb_size,2)*math.log(10000.0)/emb_size)
        # print(torch.arange(0,emb_size,2))
        """
        tensor([  0,   2,   4,   6,   8,  10,  12,  14,  16,  18,  20,  22,  24,  26,
         28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,  52,  54,
         56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,  78,  80,  82,
         84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108, 110,
        112, 114, 116, 118, 120, 122, 124, 126])
        """
        # print(-torch.arange(0,emb_size,2))
        """
        tensor([   0,   -2,   -4,   -6,   -8,  -10,  -12,  -14,  -16,  -18,  -20,  -22,
         -24,  -26,  -28,  -30,  -32,  -34,  -36,  -38,  -40,  -42,  -44,  -46,
         -48,  -50,  -52,  -54,  -56,  -58,  -60,  -62,  -64,  -66,  -68,  -70,
         -72,  -74,  -76,  -78,  -80,  -82,  -84,  -86,  -88,  -90,  -92,  -94,
         -96,  -98, -100, -102, -104, -106, -108, -110, -112, -114, -116, -118,
        -120, -122, -124, -126])
        """
        # print(math.log(10000.0)) #9.210340371976184

        # print((-torch.arange(0,emb_size,2)*math.log(10000.0)/emb_size).shape) #torch.Size([64])
        # print(-torch.arange(0,emb_size,2)*math.log(10000.0)/emb_size)
        """
        tensor([ 0.0000, -0.1439, -0.2878, -0.4317, -0.5756, -0.7196, -0.8635, -1.0074,
        -1.1513, -1.2952, -1.4391, -1.5830, -1.7269, -1.8709, -2.0148, -2.1587,
        -2.3026, -2.4465, -2.5904, -2.7343, -2.8782, -3.0221, -3.1661, -3.3100,
        -3.4539, -3.5978, -3.7417, -3.8856, -4.0295, -4.1734, -4.3173, -4.4613,
        -4.6052, -4.7491, -4.8930, -5.0369, -5.1808, -5.3247, -5.4686, -5.6126,
        -5.7565, -5.9004, -6.0443, -6.1882, -6.3321, -6.4760, -6.6199, -6.7638,
        -6.9078, -7.0517, -7.1956, -7.3395, -7.4834, -7.6273, -7.7712, -7.9151,
        -8.0590, -8.2030, -8.3469, -8.4908, -8.6347, -8.7786, -8.9225, -9.0664])
        """
        # print(torch.exp(-torch.arange(0,emb_size,2)*math.log(10000.0)/emb_size).shape) #torch.Size([64])
        # print(torch.exp(-torch.arange(0,emb_size,2)*math.log(10000.0)/emb_size))
        """
        tensor([1.0000e+00, 8.6596e-01, 7.4989e-01, 6.4938e-01, 5.6234e-01, 4.8697e-01,
        4.2170e-01, 3.6517e-01, 3.1623e-01, 2.7384e-01, 2.3714e-01, 2.0535e-01,
        1.7783e-01, 1.5399e-01, 1.3335e-01, 1.1548e-01, 1.0000e-01, 8.6596e-02,
        7.4989e-02, 6.4938e-02, 5.6234e-02, 4.8697e-02, 4.2170e-02, 3.6517e-02,
        3.1623e-02, 2.7384e-02, 2.3714e-02, 2.0535e-02, 1.7783e-02, 1.5399e-02,
        1.3335e-02, 1.1548e-02, 1.0000e-02, 8.6596e-03, 7.4989e-03, 6.4938e-03,
        5.6234e-03, 4.8697e-03, 4.2170e-03, 3.6517e-03, 3.1623e-03, 2.7384e-03,
        2.3714e-03, 2.0535e-03, 1.7783e-03, 1.5399e-03, 1.3335e-03, 1.1548e-03,
        1.0000e-03, 8.6596e-04, 7.4989e-04, 6.4938e-04, 5.6234e-04, 4.8697e-04,
        4.2170e-04, 3.6517e-04, 3.1623e-04, 2.7384e-04, 2.3714e-04, 2.0535e-04,
        1.7783e-04, 1.5399e-04, 1.3335e-04, 1.1548e-04])
        """
        
        # position_idx: torch.Size([5000, 1]) 
        # c = [1.0000e+00, 8.6596e-01, 7.4989e-01, ..., 1.3335e-04, 1.1548e-04] # torch.Size([64])
        # position_emb_fill: [[1]*c, [2]*c, [3]*c, ..., [4999]*c]
        # print(position_emb_fill.shape) #torch.Size([5000, 64])
        # print(position_emb_fill)
        """
        tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 0.0000e+00,
         0.0000e+00],
        [1.0000e+00, 8.6596e-01, 7.4989e-01,  ..., 1.5399e-04, 1.3335e-04,
         1.1548e-04],
        [2.0000e+00, 1.7319e+00, 1.4998e+00,  ..., 3.0799e-04, 2.6670e-04,
         2.3096e-04],
        ...,
        [4.9970e+03, 4.3272e+03, 3.7472e+03,  ..., 7.6950e-01, 6.6636e-01,
         5.7704e-01],
        [4.9980e+03, 4.3281e+03, 3.7480e+03,  ..., 7.6966e-01, 6.6649e-01,
         5.7716e-01],
        [4.9990e+03, 4.3290e+03, 3.7487e+03,  ..., 7.6981e-01, 6.6663e-01,
         5.7728e-01]])
        """

        pos_encoding=torch.zeros(seq_max_len,emb_size)
        # print(pos_encoding.shape) #torch.Size([5000, 128])
        pos_encoding[:,0::2]=torch.sin(position_emb_fill) #sin(torch.Size([5000, 64])) # 放入奇数位置 
        pos_encoding[:,1::2]=torch.cos(position_emb_fill) #cos(torch.Size([5000, 64])) # 放入偶数位置
        # print(torch.sin(position_emb_fill))
        # print(torch.cos(position_emb_fill))
        # register_buffer 是 PyTorch 中 nn.Module 类的一个方法，用于注册不需要梯度的参数（即不需要优化的参数）。
        # 这些参数在模型训练过程中不会改变，但在模型的前向传播过程中会被使用。
        self.register_buffer('pos_encoding',pos_encoding) # 固定参数,不需要train # 不需要train的、固定的、位置信息矩阵 #

        # 防过拟合
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):    # x: (batch_size,seq_len) #torch.Size([1, 15]) #(句子数, 句子词数)
        x=self.seq_emb(x)   # x: (batch_size,seq_len,emb_size) #torch.Size([1, 15, 128]) #(句子数, 句子词数, 词向量的维度)
        # print(self.pos_encoding.unsqueeze(0).shape) #torch.Size([1, 5000, 128])
        x=x+self.pos_encoding.unsqueeze(0)[:,:x.shape[1],:] # x: (batch_size, 【seq_len】, emb_size) #带位置信息的词向量 #按实际长度来取位置信息 #
        return self.dropout(x) #用词向量代表每个词 # 随机丢弃每个词向量中的、神经元连接

if __name__=='__main__':
    # print('len of de_vocab:', len(de_vocab)) #len of de_vocab: 19213
    emb=EmbeddingWithPosition(len(de_vocab),128) #有多少种id，就准备多少种向量 #

    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列 #15个词 #
    # print('de_tokens:', de_tokens, 'de_ids:', de_ids) #de_tokens: ['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'] de_ids: [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]

    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    # print('de_ids_tensor:', de_ids_tensor.shape) #de_ids_tensor: torch.Size([15])
    # print(de_ids_tensor.unsqueeze(0).shape) #torch.Size([1, 15])
    # print(de_ids_tensor)
    """
    tensor([   2,   21,   85,  257,   31,   87,   22,   94,    7,   16,  112, 7910,
        3209,    4,    3])
    """

    # print(de_ids_tensor.unsqueeze(0))
    """
    tensor([[   2,   21,   85,  257,   31,   87,   22,   94,    7,   16,  112, 7910,
         3209,    4,    3]])
    """
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    # print('de_ids_tensor:', de_ids_tensor.shape, 'emb_result:', emb_result.shape) #de_ids_tensor: torch.Size([15]) emb_result: torch.Size([1, 15, 128])
    # print(emb_result)
    """
    tensor([[[ 0.5343,  0.0997,  0.2833,  ...,  1.2961, -1.2359,  0.9263],
         [ 1.4525,  2.1982,  0.0000,  ...,  0.1786, -0.4752,  2.2444],
         [ 0.0000, -0.6856,  2.6001,  ...,  0.0000,  0.0000,  0.9110],
         ...,
         [-0.0000, -0.1418, -2.7486,  ...,  0.8717, -2.2171,  0.6839],
         [ 1.4487,  1.2307,  0.6357,  ...,  0.3821,  0.6519,  1.7046],
         [-0.2002, -0.0163, -1.0771,  ...,  0.0000, -0.6191,  2.1104]]],
       grad_fn=<MulBackward0>)
    """