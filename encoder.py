'''
encoder编码器,输入词id序列,输出每个词的编码向量(输入输出1:1)
'''
from torch import nn 
import torch 
from encoder_block import EncoderBlock
from emb import EmbeddingWithPosition
from dataset import de_preprocess,train_dataset,de_vocab,PAD_IDX
from config import DEVICE

class Encoder(nn.Module):
    def __init__(self,vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.emb=EmbeddingWithPosition(vocab_size,emb_size,dropout,seq_max_len)

        self.encoder_blocks=nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(emb_size,q_k_size,v_size,f_size,head))

    def forward(self,x): # x:(batch_size,seq_len)
        pad_mask=(x==PAD_IDX).unsqueeze(1) # pad_mask:(batch_size,1,seq_len) #影响score_i（即：v_i向量的系数）。 #每一个样本，具有一个注意力掩码 #
        pad_mask=pad_mask.expand(x.shape[0],x.shape[1],x.shape[1]) # pad_mask:(batch_size,seq_len,seq_len)
        # print("pad_mask.shape:",pad_mask.shape) #pad_mask.shape: torch.Size([2, 15, 15])
        # print(pad_mask)
        """
        tensor([[[False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False]],

        [[False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True],
         [False, False, False, False, False, False, False, False, False, False,
           True,  True,  True,  True,  True]]])
        """
        pad_mask=pad_mask.to(DEVICE)

        x=self.emb(x)
        for block in self.encoder_blocks:
            x=block(x,pad_mask) # x:(batch_size,seq_len,emb_size)
        return x
    
if __name__=='__main__':
    # 取2个de句子转词ID序列
    de_tokens1,de_ids1=de_preprocess(train_dataset[0][0]) #第一个德语句子
    de_tokens2,de_ids2=de_preprocess(train_dataset[1][0]) #第二个德语句子
    # print(de_tokens1, de_ids1)
    # print(de_tokens2, de_ids2)
    """
    ['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'] [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    ['<bos>', 'Mehrere', 'Männer', 'mit', 'Schutzhelmen', 'bedienen', 'ein', 'Antriebsradsystem', '.', '<eos>'] [2, 84, 31, 10, 847, 2208, 15, 8269, 4, 3]
    """

    # 组成batch并padding对齐
    if len(de_ids1)<len(de_ids2):
        de_ids1.extend([PAD_IDX]*(len(de_ids2)-len(de_ids1)))
    elif len(de_ids1)>len(de_ids2):
        de_ids2.extend([PAD_IDX]*(len(de_ids1)-len(de_ids2)))
    
    """
    [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    [2, 84, 31, 10, 847, 2208, 15, 8269, 4, 3, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]
    """
    
    # print(de_ids1)
    # print(de_ids2)

    """补齐之后的词id序列: 
    [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    [2, 84, 31, 10, 847, 2208, 15, 8269, 4, 3, 1, 1, 1, 1, 1]
    """

    batch=torch.tensor([de_ids1,de_ids2],dtype=torch.long).to(DEVICE)
    # print('batch:', batch.shape) #batch: torch.Size([2, 15])

    # Encoder编码
    encoder=Encoder(vocab_size=len(de_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    z=encoder.forward(batch)
    # print('encoder outputs:', z.shape) #torch.Size([2, 15, 128])
    # 盯着你这个隐层表示z，下一步decoder去做生成 #