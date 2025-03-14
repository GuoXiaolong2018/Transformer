'''
德语->英语翻译数据集
参考: https://pytorch.org/tutorials/beginner/translation_transformer.html
'''

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

# 下载翻译数据集
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
train_dataset = list(Multi30k(split='train', language_pair=('de', 'en')))
# print('train dataset:',train_dataset)
#  ('Frau in Schwarz arbeitet in einem Fischmarkt.', 'Woman in black working at a fish market.'),

# 创建分词器 #按空格分词 #
de_tokenizer=get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer=get_tokenizer('spacy', language='en_core_web_sm')

# 生成词表 #认识哪些英语单词和德语单词。 #
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3     # 特殊token
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

de_tokens=[] # 德语token列表
en_tokens=[] # 英语token列表
for de,en in train_dataset:
    de_tokens.append(de_tokenizer(de)) # 将句子分词
    en_tokens.append(en_tokenizer(en)) # 将句子分词

# print(en_tokens)
# ['A', 'man', 'smiling', 'into', 'the', 'camera', 'while', 'holding', 'a', 'decorative', 'plate', 'and', 'a', 'ink', 'pen'],

#词序列 -> 词id序列 #
de_vocab=build_vocab_from_iterator(de_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 德语token词表 #特殊词往前站 #
de_vocab.set_default_index(UNK_IDX) #对不认识的词设置为UNK_IDX #默认的词id # 
en_vocab=build_vocab_from_iterator(en_tokens,specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],special_first=True) # 英语token词表
en_vocab.set_default_index(UNK_IDX)
# print('en vocab:', en_vocab) #en vocab: Vocab()
# print('en vocab:', len(en_vocab)) #en vocab: 10839 #包含4个特殊词 #

# 句子特征预处理
# 德语句子 -> 词id序列
def de_preprocess(de_sentence):
    tokens=de_tokenizer(de_sentence) # 分词
    tokens=[BOS_SYM]+tokens+[EOS_SYM] # 句子前后添加开始词、结束词。 #
    ids=de_vocab(tokens) # 词表转id
    return tokens,ids

def en_preprocess(en_sentence):
    """
    对英文句子进行预处理。
    
    Args:
        en_sentence (str): 需要进行预处理的英文句子。
    
    Returns:
        tuple: 包含两个元素的元组，分别是预处理后的token列表和对应的id列表。
            - tokens (list of str): 预处理后的token列表。
            - ids (list of int): 预处理后的token对应的id列表。
    
    """
    tokens=en_tokenizer(en_sentence)
    tokens=[BOS_SYM]+tokens+[EOS_SYM]
    ids=en_vocab(tokens)
    return tokens,ids

if __name__ == '__main__':
    # 词表大小
    # print('de vocab:', len(de_vocab)) #de vocab: 19213 #包含4个特殊词
    # print('en vocab:', len(en_vocab)) #en vocab: 10839 #包含4个特殊词

    # print('de vocab:', de_vocab.get_itos())
    # print('en vocab:', en_vocab.get_itos())
    # en_vocab_list = en_vocab.get_itos()
    # sentenct = " ".join(en_vocab_list)
    # tokens, tokens_ids = en_preprocess(sentenct)
    # for i in range(50, 60):
    #     print("{}\t{}".format(tokens[i],tokens_ids[i]))
    """
    down    41
    walking 42
    -       43
    front   44
    her     45
    holding 46
    water   47
    by      48
    The     49
    up      50
    """
    # print('en preprocess:',*en_preprocess(sentenct))
    # sub_sentence = sentenct[]
    # print(type(en_vocab_list)) #<class 'list'>
    # 特征预处理
    de_sentence,en_sentence=train_dataset[0]
    # print('de sentence:',de_sentence) #de sentence: Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.
    # print('en sentence:',en_sentence) #en sentence: Two young, White males are outside near many bushes.

    # print('de sentence:', de_preprocess(de_sentence)) #(['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'], [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3])
    # print('en preprocess:', en_preprocess(en_sentence)) #(['<bos>', 'Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.', '<eos>'], [2, 19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5, 3])

    # 例子：token1, token2, token3 = tokens  # 手动解包
    # print('de preprocess:',*de_preprocess(de_sentence)) #['<bos>', 'Zwei', 'junge', 'weiße', 'Männer', 'sind', 'im', 'Freien', 'in', 'der', 'Nähe', 'vieler', 'Büsche', '.', '<eos>'] [2, 21, 85, 257, 31, 87, 22, 94, 7, 16, 112, 7910, 3209, 4, 3]
    # print('en preprocess:',*en_preprocess(en_sentence)) #['<bos>', 'Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.', '<eos>'] [2, 19, 25, 15, 1169, 808, 17, 57, 84, 336, 1339, 5, 3]