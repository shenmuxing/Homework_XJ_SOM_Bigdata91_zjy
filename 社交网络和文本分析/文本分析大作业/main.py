"""
这是主程序
"""
# coding:utf8
import os
import torch
from models import Shenmuxing_RNNLSTMModel

from dltools.config_text_data_common import opt
from dltools.text_data_processing import get_Marx_data
from dltools.rnn_train_eval_common import MarxNetTrain
from torch.utils.data import DataLoader


if __name__ == '__main__':

    opt.batch_size = 64
    opt.maxlen = 125
    opt.prefix_words = u'''图奥出身的那支民族在森林里、高地上游荡，他们不知道也不歌颂大海。不过，图奥不和他们一起生活，而是独自住在一座名叫米斯林的湖附近，有时在湖畔的林中打猎，有时在岸边用他以熊筋和木头做成的简陋竖琴弹奏乐曲。众人听说他那粗犷的歌声含有力量，纷纷从远近前来听他弹唱，但是图奥不再歌唱，动身去了偏僻荒凉的地方。他在那里见识了诸多新奇的事物，并且从流浪的诺多族那里学到了他们的语言与传承学识，但他命中注定不会永远留在那片森林中。''' #
    opt.start_words = u'刚多林的陷落'  # 题目
    opt.lr_decay = 1
    opt.lr = 0.001
    opt.clipping_grad = True
    opt.category = 'Marx'
    opt.save_freq = 20
    opt.print_freq = 40
    opt.iter_consecutive = True
    opt.jieba_rate=0.01 # jieba分词以后，保留的最高0.2%频率的词，最后保留下来了80，000个词，如果降低比率应该会更好
    opt.data_path = os.path.join(os.path.dirname(__file__), 'data/Chinese-data.txt')
    opt.pickle_path = os.path.join(os.path.dirname(__file__), 'data/all_data.npz')

    # 获取数据
    data, char_to_idx, idx_to_char = get_Marx_data(opt)

    data = torch.from_numpy(data).long()  # 转化为t.tensor, size(num_samples, maxlen)

    train_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    # 模型定义: len(char_to_idx)是总的不同字的个数，作为类别数，128是表示每个字的向量长度，256是隐藏层的维度
    model = Shenmuxing_RNNLSTMModel(len(char_to_idx), 64, 128)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    model = MarxNetTrain(model=model, optimizer=optimizer, dataloader=train_loader, idx_to_char=idx_to_char,
                      char_to_idx=char_to_idx, max_epoch=50)
    
    # 模型验证


