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
    """加载Chinese-data.txt 返回可以训练的数据类型，并保存。
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
    if os.path.exists(opt.data_path):
        with open(opt.data_path,"r",encoding="utf-8") as txt:
            corpus_chars=txt.read()
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

