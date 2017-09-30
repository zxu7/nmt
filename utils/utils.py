import os
import glob
import jieba
import re
import time
import numpy as np
from bs4 import BeautifulSoup, SoupStrainer
from keras.preprocessing.sequence import pad_sequences


def load_sgm(fn):
    strainer = SoupStrainer('seg')
    with open(fn, 'r') as f:
        data = BeautifulSoup(f.read(), 'lxml', parse_only=strainer)
    out = [str(x.string).strip( ) for x in data.find_all('seg')]
    return out


def tokenize(x, cn_tok='jieba'):
    if cn_tok == 'jieba':
        out = list(jieba.cut(x))
        return [x for x in out if x]
    elif cn_tok == 'ltp':
        pass


# TODO: better english-chinese tokenization: english should match chinese tokenization style
class Preprocess(object):
    def __init__(self, use_tok='jieba'):
        """
        :param datas: [en_data, cn_data]
        :param use_tok: 'jieba' or 'ltp'
        """
        self.en_data = None
        self.cn_data = None
        self.use_tok = use_tok
        self.en_tok = None
        self.cn_tok = None
        self.en_dict = None
        self.cn_dict = None

    def en_tokenize(self, x):
        out = re.findall(r'[.,?!:()]+|[0-9.]+|[a-zA-Z0-9\-()]+|[\'\w\".]+', x)
        return [x.lower() for x in out]

    def cn_tokenize(self, x):
        if self.use_tok == 'jieba':
            return list(jieba.cut(x))
        elif self.use_tok == 'ltp':
            raise NotImplementedError

    def fit(self, datas):
        self.en_data = datas[0]
        self.cn_data = datas[1]
        self.en_tok = [self.en_tokenize(x) for x in self.en_data]
        self.cn_tok = [self.cn_tokenize(x) for x in self.cn_data]
        self.en_maxlen = max([len(x) for x in self.en_tok])
        self.cn_maxlen = max([len(x) for x in self.cn_tok])
        a1 = []
        a2 = []
        [a1.extend(x) for x in self.en_tok]
        [a2.extend(x) for x in self.cn_tok]
        en_keys = ["<PAD>",] + list(set(a1))
        cn_keys = ["<PAD>",] + list(set(a2))
        # en_keys.insert(0, "<PAD>")
        # cn_keys.insert(0, "<PAD>")
        self.en_dict = {k:i for i,k in enumerate(en_keys)}
        self.cn_dict = {k:i for i,k in enumerate(cn_keys)}

    def get_tokens(self):
        return self.en_tok, self.cn_tok

    def get_ids(self):
        en = []
        cn = []
        for x in self.en_tok:
            en.append([self.en_dict[x1] for x1 in x])

        for x in self.cn_tok:
            cn.append([self.cn_dict[x1] for x1 in x])
        return en, cn


class WordVector(object):
    def __init__(self, en_dict, cn_dict):
        """
        :param en_dict: word: id
        :param cn_dict: word: id
        """
        self.en_dict = en_dict
        self.cn_dict = cn_dict
        self.en_wv_path = None
        self.cn_wv_path = None

    def to_matrix(self, en_wv_path, cn_wv_path):
        en_wv_dict = self._wv2dict(en_wv_path)
        cn_wv_dict = self._wv2dict(cn_wv_path)
        en_mat = self._dict2mat(self.en_dict, en_wv_dict)
        cn_mat = self._dict2mat(self.cn_dict, cn_wv_dict)
        return en_mat, cn_mat

    # TODO: faster for-loop
    def _wv2dict(self, fn):
        now = time.time()
        glove_dict = {}
        glove = []
        with open(fn, 'r') as f:
            for line in f:
                glove.append(line.strip('\n'))
        for x in glove:
            k, v = self._row2array(x)
            glove_dict[k] = v
        print("DEBUG: convert wv to dict take ", time.time()-now)
        return glove_dict

    def _dict2mat(self, id_dict, wv_dict):
        now = time.time()

        n = len(id_dict)
        d = len(wv_dict[list(wv_dict.keys())[0]])
        out = np.zeros((n,d))

        # def fill_out(k):
        #     v = wv_dict.get(k, np.random.normal(scale=0.6, size=d))
        #     out[id_dict[k], :] = v
        # [(lambda x: fill_out(x))(key) for key in id_dict.keys()]

        for key in id_dict.keys():
            v = wv_dict.get(key, np.random.normal(scale=0.6, size=(d)))
            out[id_dict[key], :] = v

        print("DEBUG: convert dicts to matrix take {}".format(time.time()-now))
        return out

    @staticmethod
    def _row2array(x):
        x1 = x.split(' ')
        array = x1[1:]
        array = np.array([float(a) for a in array], dtype=np.float32)
        return x1[0], array
