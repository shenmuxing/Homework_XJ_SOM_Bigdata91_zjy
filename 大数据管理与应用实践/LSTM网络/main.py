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

    opt.batch_size = 32
    opt.maxlen = 125
    opt.prefix_words = u'''材料1进入新时代，我国面临更为严峻的国家安全形势，外部压力前所未有，传统安全威胁和非传统安全威胁相互交织，“黑天鹅”、“灰犀牛”事件时有发生。同形势任务要求相比，我国维护国家安全能力不足，应对各种重大风险能力不强，维护国家安全的统筹协调机制不健全。党中央强调，国泰民安是人民群众最基本、最普遍的愿望。必须坚持底线思维、居安思危、未雨绸缪，坚持国家利益至上，以人民安全为宗旨，以政治安全为根本，以经济安全为基础，以军事、科技、文化、社会安全为保障，以促进国际安全为依托，统筹发展和安全，统筹开放和安全，统筹传统安全和非传统安全，统筹自身安全和共同安全，统筹维护国家安全和塑造国家安全。摘自《中共中央关于党的百年奋斗重大成就和历史经验的决议》
    材料2“黑天鹅”事件通常指现实生活中出现的“出乎预料”的小概率事件。在特定时间内发生的可能性相对较低的小概率事件，广泛存在于自然、经济、政治等各个领域，具有偶发性、难以预测性等特征。小概率事件虽然发生概率小，但并非零概率事件，若从长时段来看，只要具备相关因素和条件，就可能会发生。小概率事件的影响不局限于一时一地，一旦发生，就可能会形成多米诺骨牌效应，导致系统性风险，给整个人类社会发展带来深远影响。习近平总书记在庆祝中国共产党成立100周年大会上的讲话中指出，新的征程上，我们必须增强忧患意识、始终居安思危，深刻认识我国社会主要矛盾变化带来的新特征新要求，深刻认识错综复杂的国际环境带来的新矛盾新挑战，敢于斗争，善于斗争，逢山开道、遇水架桥，勇于战胜一切风险挑战。应对我国发展环境深刻复杂变化，特别是其中隐藏的重大风险挑战，是向着全面建成社会主义现代化强国目标迈进的必然要求。我国发展具有多方面优势和条件，但我国发展不平衡不充分问题仍然突出，重难点领域关键环节改革任务仍然艰巨。发展环境的深刻复杂变化，既要求我们牢牢把握机遇发展自己，又要求我们树立辩证思维，提高驾驭复杂局面、处理复杂问题的本领。''' # 材料,来源http://www.kaoyan365.cn/zhengzhi/zhenti/304132.html
    opt.start_words = u'面对复杂局面应如何运用好辩证思维?'  # 题目
    opt.lr_decay = 1
    opt.lr = 0.001
    opt.clipping_grad = True
    opt.category = 'Marx'
    opt.save_freq = 20
    opt.print_freq = 40
    opt.iter_consecutive = True
    opt.jieba_rate=0.002 # jieba分词以后，保留的最高0.2%频率的词，最后保留下来了80，000个词，如果降低比率应该会更好
    opt.data_path = os.path.join(os.path.dirname(__file__), 'data/Marx/')
    opt.pickle_path = os.path.join(os.path.dirname(__file__), 'data/Marx/all_data.npz')

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


