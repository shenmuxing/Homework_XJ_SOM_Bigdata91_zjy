"""
数据处理相关的函数
"""

import math
import os
import random
import time
import zipfile
import copy

import torch
from torch import nn
import torch.nn.functional as F
from dltools.config_text_data_common import opt
import tqdm
import jieba
import gc

#--------------Marx------------------------
def MarxNetTrain(model, optimizer, dataloader, idx_to_char, char_to_idx, **kwargs):
    opt._parse(kwargs)
    model.to(opt.device)
    criterion = opt.criterion

    state = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay)  # 设置学习率下降策略

    if os.path.exists('checkpoints') == False:
        os.mkdir('checkpoints')

    # train
    print("training on ", opt.device)

    with open(model.model_name + '_' + opt.category + '.txt', encoding='utf-8', mode='w') as fp:

        for epoch in range(opt.max_epoch):
            train_loss_sum, n, start = 0.0, 0, time.time()

            model.train()

            for ii, data_ in enumerate(dataloader):
                if (ii + 1 == len(dataloader)) & (data_.shape[0] != opt.batch_size):  # 当最后一个batch的样本个数小于batch_size时，舍弃最后一个batch
                    break

                if (state is not None) & opt.iter_consecutive==True:
                    # 使用detach函数从计算图分离隐藏状态, 这是为了
                    # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                    if isinstance(state, tuple):  # LSTM, state:(h, c)
                        state = (state[0].detach(), state[1].detach())
                    else:
                        state = state.detach()
                else:
                    state = None
                # forward
                data_ = data_.to(opt.device)  # 第1维是batch_size, 第2维是每首诗长度

                X, Y = data_[:, :-1], data_[:, 1:]  # X是前n-1列，Y是后n-1列, X和Y的形状是(batch_size, num_steps)
                (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)
                
                del data_,X
                gc.collect()
                torch.cuda.empty_cache()
                
                optimizer.zero_grad()
                y = torch.transpose(Y, 0, 1).contiguous().view(-1)

                loss = criterion(output, y.long())# 内存没那么富裕，换成普通的试试
                
                # loss = criterion(output,y)
                loss.backward()

                # 梯度裁剪
                if opt.clipping_grad == True:
                    params = [p for p in model.parameters()]
                    grad_clipping(params, opt.clipping_theta, opt.device)
                optimizer.step()



                train_loss_sum += loss.item() * y.shape[0]
                n += y.shape[0]
                del loss
                gc.collect()
                torch.cuda.empty_cache()
                
                if (ii + 1) % opt.print_freq == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                          .format(epoch, int(n / (opt.maxlen-1)), len(dataloader.dataset),
                                  100. * (n / (opt.maxlen-1)) / len(dataloader.dataset),
                                  train_loss_sum / n))

            if (epoch + 1) % opt.save_freq == 0:
                torch.save(model.state_dict(), f='checkpoints\\' + model.model_name + '_' + str(epoch) + opt.category + '.pth')
            
            try:
                perplexity = math.exp(train_loss_sum / n)
            except OverflowError:
                perplexity = float('inf')

            # 生成语言模型的结果显示
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch, perplexity, time.time() - start))
            fp.write(u'第{}epoch\n'.format(epoch))

            gen_Marx_list = Marx_val_generate(model, opt.start_words, idx_to_char, char_to_idx, opt.device, opt.prefix_words)

            gen_Marx = ''.join(gen_Marx_list)
            print(gen_Marx)
            fp.write(gen_Marx + '\n')
            fp.write('\n')

            scheduler.step()  # 更新学习率

        return model


def Marx_val_generate(model, start_words, idx_to_char, char_to_idx, device, prefix_words=None):
    model.eval()
    results = [i for i in jieba.cut(start_words)]
    start_word_len = len(results)

    # 手动设置第一个词为空格
    X = torch.tensor([char_to_idx[' ']], device=device).view(1, 1).long()
    state = None

    # 此循环主要是获得state数据，从而得到意境
    if prefix_words:
        # 需要先将prefix_words分词
        prefix_words=[i for i in jieba.cut(prefix_words)]
        for word in prefix_words:
            _, state = model(X, state)
            try:
              X = X.data.new([char_to_idx[word]]).view(1, 1)  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致
            except KeyError as e:
              # 可能出现不认识的字或词，此时跳过
              continue
              

    
    for i in range(opt.max_gen_len):
        Y, state = model(X, state)

        if i < start_word_len:
            w = results[i]
            X = X.data.new([char_to_idx[w]]).view(1, 1)
        else:
            # top_index = Y.data[0].topk(1)[1][0].item()
            # w = idx_to_char[top_index]
            top_index = Y.argmax(dim=1).long().item()
            w = idx_to_char[top_index]

            results.append(w)
            X = X.data.new([top_index]).view(1, 1)

    model.train()
    return results


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)