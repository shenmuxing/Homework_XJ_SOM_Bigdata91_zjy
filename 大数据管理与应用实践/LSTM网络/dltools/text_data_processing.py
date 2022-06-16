import os
import random
import zipfile
import torch
import torch.nn.functional as F
import copy
import jieba
from collections import Counter
import re
import numpy as np
#---------------处理马原相关文章------------------------
def get_Marx_data(opt):
    """加载马原相关文章数据集
    输入：class Config类，详见本库下的config_text_data_commmon.py，在opt类中
    使用jieba把文章进行分词
    """
    if os.path.exists(opt.pickle_path):
        print("直接加载数据...")
        data = np.load(opt.pickle_path, allow_pickle=True)
        data, char_to_idx, idx_to_char = data['data'], data['char_to_idx'].item(), data['idx_to_char'].item()
        print("获得了",data.shape,"的漂亮数据")
        return data, char_to_idx, idx_to_char
    print("自己组合数据...")
    corpus_chars=""
    for file1 in os.listdir(opt.data_path):
      if not os.path.isdir(opt.data_path+"/"+file1):
        with zipfile.ZipFile(opt.data_path+file1) as thezip:
            for info in thezip.namelist():
                if info.endswith(".txt"):
                    with thezip.open(info) as txt:
                        corpus_chars=corpus_chars+txt.read().decode("utf-8")
    
    corpus_chars=corpus_chars.replace("\n"," ").replace("\r"," ")

    print("使用jieba分词...")
    corpus_list=jieba.lcut(corpus_chars)#jieba分词，结果是list
    corpus_list_no1=[i for i in corpus_list if len(i)>1 and re.match(r"[\u4e00-\u9fa5]",i)]#删掉不是中文以及只有一个的词
    #统计最高的词频，只保留opt.jieba_rate比率的词
    temp=Counter(corpus_list_no1).most_common(round(opt.jieba_rate*len(corpus_list_no1)))
    corpus_list=[temp[i][0] for i in range(len(temp))]
    chars = set(corpus_list)
    print("jieba总共分出来了",len(chars),"个词")
    # print("字一共有:",len(set(corpus_chars)),"个")
    chars = set(corpus_chars).union(chars)
    # 整个文本所有的入选词汇，包括jieba分出来的以及文中的所有字，用jieba.add_word功能权重加到非常大
    # 然后再用jieba分词分成想要的效果,比如"中华人民共和国成立70周年"
    # 原来分成"中华人民共和国/成立/70/周年"
    # 现在分成"中华人民共和国/成/立/70/周/年"(只有中华人民共和国是高频词)
    for char in chars:
        jieba.add_word(char,10000000000000)
        if len(char)>1:
            jieba.suggest_freq(char, tune=True)
    corpus_list=jieba.lcut(corpus_chars,HMM=False)
    chars = set(corpus_list)
    print("分词完成,最终入选的字和词有",len(chars),"个...")
    char_to_idx = {char: i for i, char in enumerate(chars)}  # 生成char与index的对应关系
    idx_to_char = {i: char for char, i in list(char_to_idx.items())}
    corpus_indices = [char_to_idx[char] for char in corpus_list]  # 将文章中的字符转化为数字


    # print("使用jieba分词...")
    
    # corpus_list=[i for i in jieba.cut(corpus_chars) if len(i)>=1]
    # temp=Counter(corpus_list).most_common(round(opt.jieba_rate*len(corpus_list)))
    # corpus_list=[temp[i][0] for i in range(len(temp))]
    # print("分词完成...")
    # chars = set(corpus_list)
    # print("jieba总共分出来了",len(chars),"个词")
    # chars = set(corpus_chars).union(chars)
    # char_to_idx = {char: i for i, char in enumerate(chars)}  # 生成char与index的对应关系
    # idx_to_char = {i: char for char, i in list(char_to_idx.items())}
    # corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 将文章中的字符转化为数字


    data_vec = np.ones(int(opt.batch_size * opt.maxlen * np.ceil(len(corpus_indices) / (opt.batch_size * opt.maxlen) ))) * char_to_idx[' ']
    data_vec[:len(corpus_indices)] = np.asarray(corpus_indices)

    data_mat = data_vec.reshape(opt.batch_size, -1)

    n_batch_row = int(data_mat.shape[1] / opt.maxlen)

    data = np.zeros((opt.batch_size * n_batch_row, opt.maxlen))
    for ii in range(n_batch_row):
        data[ii * opt.batch_size: (ii+1) * opt.batch_size, :] = data_mat[:, ii * opt.maxlen:(ii+1) * opt.maxlen]

    # 保存成二进制文件
    np.savez_compressed(opt.pickle_path,
                        data=data,
                        char_to_idx=char_to_idx,
                        idx_to_char=idx_to_char)
    print("获得了",data.shape,"维度的漂亮数据...")
    print("自己组合数据完成！祝您生活愉快...")
    return data, char_to_idx, idx_to_char

# --------------------处理歌词----------------------------
def get_lyrics_data(opt):
    """加载周杰伦歌词数据集"""
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path, allow_pickle=True)
        data, char_to_idx, idx_to_char = data['data'], data['char_to_idx'].item(), data['idx_to_char'].item()
        return data, char_to_idx, idx_to_char


    with zipfile.ZipFile(opt.data_path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    chars = set(corpus_chars)
    char_to_idx = {char: i for i, char in enumerate(chars)}  # 生成char与index的对应关系
    idx_to_char = {i: char for char, i in list(char_to_idx.items())}

    corpus_indices = [char_to_idx[char] for char in corpus_chars]  # 将歌词中的字符转化为数字

    data_vec = np.ones(int(opt.batch_size * opt.maxlen * np.ceil(len(corpus_indices) / (opt.batch_size * opt.maxlen) ))) * char_to_idx[' ']
    data_vec[:len(corpus_indices)] = np.asarray(corpus_indices)

    data_mat = data_vec.reshape(opt.batch_size, -1)

    n_batch_row = int(data_mat.shape[1] / opt.maxlen)

    data = np.zeros((opt.batch_size * n_batch_row, opt.maxlen))
    for ii in range(n_batch_row):
        data[ii * opt.batch_size: (ii+1) * opt.batch_size, :] = data_mat[:, ii * opt.maxlen:(ii+1) * opt.maxlen]

    # 保存成二进制文件
    np.savez_compressed(opt.pickle_path,
                        data=data,
                        char_to_idx=char_to_idx,
                        idx_to_char=idx_to_char)

    return data, char_to_idx, idx_to_char








### ---------------处理诗词---------------------------

# coding:utf-8
import sys
import os
import json
import re
import numpy as np


def poem_parseRawData(author=None, constrain=None, src='./chinese-poetry/json/simplified', category="poet.tang"):
    """
    code from https://github.com/justdark/pytorch-poetry-gen/blob/master/dataHandler.py
    处理json文件，返回诗歌内容
    @param: author： 作者名字
    @param: constrain: 长度限制
    @param: src: json 文件存放路径
    @param: category: 类别，有poet.song 和 poet.tang

    返回 data：list
        ['床前明月光，疑是地上霜，举头望明月，低头思故乡。',
         '一去二三里，烟村四五家，亭台六七座，八九十支花。',
        .........
        ]
    """

    def sentenceParse(pdata):
        # para 形如 "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，
        # 生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）
        # 好是去塵俗，煙花長一欄。"
        result, number = re.subn(u"（.*）", "", pdata)  # u/U:表示unicode字符串，对中文必须表明所需编码, 否则一旦编码转换就会出现乱码。
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        result, number = re.subn(r"[\d-]+", "", result)
        # r = ""
        # for s in result:
        #     if s not in set('0123456789-'):
        #         r += s
        r, number = re.subn(u"。。", u"。", result)
        return r

    def handleJson(file):
        # print file
        rst = []
        # data = json.loads(open(file, encoding='utf-8').read())
        with open(file, mode='r', encoding='utf-8') as fp:
            data = json.load(fp)
        for poetry in data:

            if (author is not None and poetry.get("author") != author): # 选定特定作者的诗句，不是该作者的跳过
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:  # 选定每句诗词的字数，不是这个字数的跳过
                        flag = True
                        break
                if flag:
                    break
            if flag:
                continue
            # for sentence in poetry.get("paragraphs"):
            #     pdata += sentence  # 将这一首诗拼接为一整个字符串
            pdata = ''.join(p)
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src + filename))
    return data


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break
    # 若每首诗样本存在其它的维度，需要将其它维度也考虑进去，即x的shape为(num_samples, maxlen, other_dim1, other_dim2)
    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_poem_data(opt):
    """
    @param opt 配置选项 Config对象
    @return char_to_idx: dict,每个字对应的序号，形如u'月'->100
    @return idx_to_char: dict,每个序号对应的字，形如'100'->u'月'
    @return data: numpy数组，每一行是一首诗对应的字的下标
    """
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path, allow_pickle=True)
        data, char_to_idx, idx_to_char = data['data'], data['char_to_idx'].item(), data['idx_to_char'].item()
        return data, char_to_idx, idx_to_char

    # 如果没有处理好的二进制文件，则处理原始的json文件
    data = poem_parseRawData(opt.author, opt.constrain, opt.data_path, opt.category)
    words = {_word for _sentence in data for _word in _sentence}  # 构建
    char_to_idx = {_word: _ix for _ix, _word in enumerate(words)}
    char_to_idx['<EOP>'] = len(char_to_idx)  # 终止标识符 9216
    char_to_idx['<START>'] = len(char_to_idx)  # 起始标识符 9217
    char_to_idx[' '] = len(char_to_idx)  # 空格 9218
    idx_to_char = {_ix: _word for _word, _ix in list(char_to_idx.items())}

    # 为每首诗歌加上起始符和终止符
    for i in range(len(data)):
        data[i] = ["<START>"] + list(data[i]) + ["<EOP>"]

    # 将每首诗歌保存的内容由‘字’变成‘数’
    # 形如[春,江,花,月,夜]变成[1,2,3,4,5]
    new_data = [[char_to_idx[_word] for _word in _sentence]
                for _sentence in data]

    # 诗歌长度不够opt.maxlen的在前面补空格，超过的，删除末尾的
    pad_data = pad_sequences(new_data,
                             maxlen=opt.maxlen,
                             padding=opt.padding,
                             truncating=opt.truncating,
                             value=len(char_to_idx) - 1)

    # 保存成二进制文件
    np.savez_compressed(opt.pickle_path,
                        data=pad_data,
                        char_to_idx=char_to_idx,
                        idx_to_char=idx_to_char)
    return pad_data, char_to_idx, idx_to_char






# -------------------------------备份lyrics程序----------------------------------


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    # scatter(dim, index, src)的参数有3个
    # dim：沿着哪个维度进行索引
    # index：用来scatter的元素索引
    # src：用来scatter的源元素，可以是一个标量或一个张量
    # 官方文档给出了3维张量的具体操作说明
    # self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    # self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    seq_len = X.shape[1]
    return [one_hot(X[:, i], n_class) for i in range(seq_len)]



class text_data_iter_random1(object):
    def __init__(self, corpus_indices, batch_size, num_steps, device=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.corpus_indices = corpus_indices
        self.batch_size = batch_size
        self.num_steps = num_steps

        # num_steps是每个样本所包含的时间步数，减1是因为输出的索引x是相应输入的索引y加1
        self.num_examples = (len(corpus_indices) - 1) // num_steps  # 样本的个数
        self.epoch_size = num_examples // batch_size  # batch的个数


        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self):
        example_indices = list(range(self.num_examples))  # 样本index
        random.shuffle(example_indices)  # 随机打乱样本index

        while self.i < self.epoch_size:
            # 每次读取batch_size个随机样本
            j = copy.deepcopy(self.i)
            j = j * self.batch_size
            batch_indices = example_indices[j: j + self.batch_size]
            X = [self._data(k * num_steps) for k in batch_indices]
            Y = [self._data(k * num_steps + 1) for k in batch_indices]
            self.i += 1
            yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32,
                                                                                    device=device)


    # 返回从pos开始的长为num_steps的序列
    def _data(self, pos):
        return self.corpus_indices[pos: pos + self.num_steps]



# def text_data_iter_random(corpus_indices, batch_size, num_steps, device=None):
#     # 以迭代器的方式返回随机采样后的数据和标签，每次读取batch_size个样本，这里的数据和标签都是index
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # num_steps是每个样本所包含的时间步数，减1是因为输出的索引x是相应输入的索引y加1
#     num_examples = (len(corpus_indices) - 1) // num_steps  # 样本的个数
#     epoch_size = num_examples // batch_size  # batch的个数
#     example_indices = list(range(num_examples))  # 样本index
#     random.shuffle(example_indices)  # 随机打乱样本index
#
#     # 返回从pos开始的长为num_steps的序列
#     def _data(pos):
#         return corpus_indices[pos: pos + num_steps]
#
#     for i in range(epoch_size):
#         # 每次读取batch_size个随机样本
#         i = i * batch_size
#         batch_indices = example_indices[i: i + batch_size]
#         X = [_data(j * num_steps) for j in batch_indices]
#         Y = [_data(j * num_steps + 1) for j in batch_indices]
#         yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


class text_data_iter_consecutive1(object):
    def __init__(self, corpus_indices, batch_size, num_steps, device=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.corpus_indices = torch.tensor(corpus_indices, dtype=torch.long, device=device)
        self.batch_len = len(corpus_indices) // batch_size  # 每个batch包含的字符数
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = (self.batch_len - 1) // num_steps  # 若时间步长设置为num_steps，可以有多少batch
        self.indices = self.corpus_indices[0: batch_size * self.batch_len].view(batch_size, self.batch_len)
        self.i = 0

    def reset(self):
        self.i = 0

    def __call__(self):
        while self.i < self.epoch_size:
            # 每次读取batch_size个随机样本
            j = copy.deepcopy(self.i)
            j = j * self.num_steps
            X = self.indices[:, j: j + self.num_steps]
            Y = self.indices[:, j + 1: j + self.num_steps + 1]
            self.i += 1
            yield X, Y





# def text_data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
#     # 以迭代器的方式返回相邻采样后的数据和标签，每次读取batch_size个样本，这里的数据和标签都是index
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
#
#     data_len = len(corpus_indices)  # 字符总长度
#     batch_len = data_len // batch_size  # 每个batch包含的字符数
#     indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
#     epoch_size = (batch_len - 1) // num_steps  # 若时间步长设置为num_steps，可以有多少batch
#     for i in range(epoch_size):
#         i = i * num_steps
#         X = indices[:, i: i + num_steps]
#         Y = indices[:, i + 1: i + num_steps + 1]
#         yield X, Y


def load_data_jay_lyrics(data_path, data_len):
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile(data_path) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:data_len]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size