import torch
from dataset import de_preprocess,train_dataset,BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,en_vocab
from config import DEVICE,SEQ_MAX_LEN

# de翻译到en
def translate(transformer,de_sentence):
    # De分词
    de_tokens,de_ids=de_preprocess(de_sentence) #把一句话（字符串）处理成
    # print("##de_tokens:", de_tokens) ###de_tokens: ['<bos>', 'Zwei', 'Männer', 'unterhalten', 'sich', 'mit', 'zwei', 'Frauen', '<eos>']
    # print("##de_ids:", de_ids) ###de_ids: [2, 21, 31, 190, 25, 10, 74, 46, 3]
    if len(de_tokens)>SEQ_MAX_LEN:
        raise Exception('不支持超过{}的句子'.format(SEQ_MAX_LEN))

    # Encoder阶段
    enc_x_batch=torch.tensor([de_ids],dtype=torch.long).to(DEVICE)      # 准备encoder输入
    # print("##enc_x_batch.shape:", enc_x_batch.shape) ##enc_x_batch.shape: torch.Size([1, 9])
    # print("##enc_x_batch:", enc_x_batch) ##enc_x_batch: tensor([[  2,  21,  31, 190,  25,  10,  74,  46,   3]])
    encoder_z=transformer.encode(enc_x_batch)    # encoder编码
    # print("##encoder_z.shape:", encoder_z.shape) ##encoder_z.shape: torch.Size([1, 9, 512])

    # Decoder阶段
    en_token_ids=[BOS_IDX] # 翻译结果
    while len(en_token_ids)<SEQ_MAX_LEN:
        dec_x_batch=torch.tensor([en_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
        decoder_z=transformer.decode(dec_x_batch,encoder_z,enc_x_batch)   # decoder解碼
        # print("##decoder_z.shape:", decoder_z.shape) ##decoder_z.shape: torch.Size([1, 1, 10837])  
        next_token_probs=decoder_z[0,dec_x_batch.shape[-1]-1,:]    # 序列下一个词的概率 #或者：decoder_z[0,len(en_token_ids)-1,:]
        # print("##next_token_probs.shape:", next_token_probs.shape) ##next_token_probs.shape: torch.Size([10837])
        # print(dec_x_batch.size()) #torch.Size([1, 1~9])
        next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
        # print("##next_token_id.shape:", next_token_id.shape) ##next_token_id.shape: torch.Size([])
        # print("##next_token_id:", next_token_id)
        """
        ##next_token_id: tensor(19)
        ##next_token_id: tensor(36)
        ##next_token_id: tensor(17)
        ##next_token_id: tensor(121)
        ##next_token_id: tensor(14)
        ##next_token_id: tensor(66)
        ##next_token_id: tensor(52)
        ##next_token_id: tensor(5)
        ##next_token_id: tensor(3)
        """
        en_token_ids.append(next_token_id) #更新

        if next_token_id==EOS_IDX:  # 结束符
            break

    # 生成翻译结果
    en_token_ids=[id for id in en_token_ids if id not in [BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX]] # 忽略特殊字符
    en_tokens=en_vocab.lookup_tokens(en_token_ids)    # 词id序列转token序列
    # print("##en_tokens:", en_tokens) ##en_tokens: ['Two', 'men', 'are', 'talking', 'with', 'two', 'women', '.']
    return ' '.join(en_tokens)


if __name__=='__main__':
    # 加载模型
    transformer=torch.load('checkpoint/model.pth', map_location=torch.device('cpu'))
    transformer.eval()
    
    en=translate(transformer,'Zwei Männer unterhalten sich mit zwei Frauen')
    # print(en) #Two men are talking with two women .

    '''
    # 测试数据
    for i in range(100):
        de,en=train_dataset[i]
        en1=translate(transformer,de)
        print('{} -> {} -> {}'.format(de,en,en1))
    '''