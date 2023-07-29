# 构建单词表，用于token和id之间的相互转化
# 出现次数小于5次的词用特殊符号<unk>代替

import numpy as np
import json
import pickle
from tqdm import tqdm

class Vocabulary():
    """单词表"""
    def __init__(self):
        self._word2id = {}
        self._id2word = {}
        self._idx = 0
        self._word = []

        # 特殊符号
        self.pad = '<pad>'  # 用于将长度补齐的标识符
        self.bos = '<bos>'  # 开始符号
        self.eos = '<eos>'  # 结束符号
        self.unk = '<unk>'  # unknown符号
        self.add_spe_sign()

    def add_word(self, word):
        '''添加单词'''
        if word not in self._word:
            self._word2id.update({word: self._idx})
            self._id2word.update({self._idx: word})
            self._word.append(word)
            self._idx += 1

    def word_to_id(self, word):
        '''把word转换成id的形式'''
        if word in self._word:
            return self._word2id[word]
        else:
            return self._word2id['<unk>']

    def id_to_word(self, id):
        '''把id的形式转换成word'''
        assert id <= self._idx, "输入的id大于最大的id"
        return self._id2word[id]

    def tokenList_to_idList(self, tokenList, fixed_len):
        '''把tokenList转换成id的形式，，同时添加上<bos>，<eos>和<pad>
        :param tokenList: 包含一个句子的token形式， 如 ["室内", "三个", "衣着", "各异", "的", "人", "坐在", "桌子", "旁", "交谈"]
        :param fixed_len: 句子的最大长度，包括<bos>和<eos>
        :return: list
        '''
        sent_len = len(tokenList)
        tok_id = [self.word_to_id(token) for token in tokenList]
        if sent_len < fixed_len:
            tok_id.insert(0, self._word2id[self.bos])
            tok_id.append(self._word2id[self.eos])
            pad_num = fixed_len - sent_len
            tok_id += [0] * pad_num
        else:
            tok_id = tok_id[:fixed_len]
            tok_id.insert(0, self._word2id[self.bos])
            tok_id.append(self._word2id[self.eos])
            sent_len = fixed_len
        sent_len += 2   # 加上开始结束符
        return tok_id, sent_len

    def idList_to_sent(self, id_List):
        '''把idList转换成sent的形式
        :param id_List: 包含一个句子的id形式，如: [1, 4, 5, 343, 4, 123, 2389 ,213, 233 ,678 ,2343 ,2, 0, 0, 0, 0, 0, 0]
                        支持格式,: list, tensor, numpy.array
        :return: 一个str句子，如: "室内三个衣着各异的人坐在桌子旁交谈"
        '''
        id_List = np.array(list(map(int, id_List)))
        word_array = np.array(self._word)
        eos_id = self._word2id[self.eos]
        eos_pos = np.where(id_List == eos_id)[0]
        if len(eos_pos >= 0):
            sent = word_array[id_List[1:eos_pos[0]]]
        else:
            sent = word_array[id_List[1:]]
        return ' '.join(sent)

    def add_spe_sign(self):
        self.add_word(self.pad)
        self.add_word(self.bos)
        self.add_word(self.eos)
        self.add_word(self.unk)

    def get_size(self):
        return self._idx

if __name__ == '__main__':
    vocab = Vocabulary()
    data_train = json.load(open('../data/train.json', 'r'))

    counter = {}
    for item in tqdm(data_train):
        sentence_token = item['caption']
        for token in sentence_token:
            counter[token] = counter.get(token, 0) + 1
    # cand_word = [token for token, f in counter.items() if f >= 5]
    print(counter['tesla'])
    input()
    cand_word = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("word (f>=5) num: "+str(len(cand_word)))

    for word in cand_word:
        vocab.add_word(word)
    print("vocab size: "+str(vocab.get_size()))

    # pickle.dump(vocab, open('../data/vocab.pkl', 'wb'))