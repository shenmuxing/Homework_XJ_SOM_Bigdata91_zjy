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

# --------------lyrics ------------------------------

def lyrics_val_generate(model, start_words, idx_to_char, char_to_idx, device, prefix_words=None):
    model.eval()
    results = list(start_words)
    start_word_len = len(start_words)

    # 手动设置第一个词为空格
    X = torch.tensor([char_to_idx[' ']], device=device).view(1, 1).long()
    state = None

    # 此循环主要是获得state数据，从而得到意境
    if prefix_words:
        for word in prefix_words:
            _, state = model(X, state)
            X = X.data.new([char_to_idx[word]]).view(1, 1)  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致

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


def LyricsNetTrain(model, optimizer, dataloader, idx_to_char, char_to_idx, **kwargs):
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

                optimizer.zero_grad()
                y = torch.transpose(Y, 0, 1).contiguous().view(-1)
                loss = criterion(output, y.long())
                loss.backward()

                # 梯度裁剪
                if opt.clipping_grad == True:
                    params = [p for p in model.parameters()]
                    grad_clipping(params, opt.clipping_theta, opt.device)
                optimizer.step()



                train_loss_sum += loss.item() * y.shape[0]
                n += y.shape[0]


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

            gen_lyrics_list = lyrics_val_generate(model, opt.start_words, idx_to_char, char_to_idx, opt.device, opt.prefix_words)

            gen_lyrics = ''.join(gen_lyrics_list)
            print(gen_lyrics)
            fp.write(gen_lyrics + '\n')
            fp.write('\n')

            scheduler.step()  # 更新学习率

        return model





def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)



# --------------poem ------------------------------

def PoemNetTrain(model, optimizer, dataloader, idx_to_char, char_to_idx, **kwargs):
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
            # for ii, data_ in tqdm.tqdm(enumerate(dataloader), desc='Epoch: {}'.format(epoch)):
                if (ii+1 == len(dataloader)) & (data_.shape[0] != opt.batch_size): # 当最后一个batch的样本个数小于batch_size时，舍弃最后一个batch
                    break

                if (state is not None) & opt.iter_consecutive == True:
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

                X, Y = data_[:, :-1], data_[:, 1:]  # X是前n-1列，Y是后n-1列, X和Y的形状是(batch_size， num_steps)
                output, state = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

                # backward
                optimizer.zero_grad()
                y = torch.transpose(Y, 0, 1).contiguous().view(-1)  # batch_size * num_steps 的向量，这样跟输出的行一一对应
                loss = criterion(output, y.long())
                loss.backward()

                # 梯度裁剪
                if opt.clipping_grad == True:
                    params = [p for p in model.parameters()]
                    grad_clipping(params, opt.clipping_theta, opt.device)
                optimizer.step()

                train_loss_sum += loss.item() * y.shape[0]  # item将标量张量转化为python number
                n += y.shape[0]

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
            print('epoch %d, perplexity %f, time %.2f sec' % ( epoch, perplexity, time.time() - start))
            fp.write(u'第{}epoch\n'.format(epoch))

            if opt.acrostic == False:
                gen_poetry_list = poem_val_generate(model, opt.start_words, idx_to_char, char_to_idx, opt.device, opt.prefix_words)
            else:
                gen_poetry_list = poem_val_acrostic_generate(model, opt.start_words, idx_to_char, char_to_idx, opt.device, opt.prefix_words)

            gen_poetry = ''.join(gen_poetry_list)
            print(gen_poetry)
            fp.write(gen_poetry + '\n')
            fp.write('\n')

            scheduler.step()  # 更新学习率

    return model


@torch.no_grad()
def poem_val_generate(model, start_words, idx_to_char, char_to_idx, device, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """
    # prefix_words 不是诗歌的组成部分，用来控制生成诗歌的意境
    model.eval()
    results = list(start_words)
    start_word_len = len(start_words)

    # 手动设置第一个词为<START>
    X = torch.tensor([char_to_idx['<START>']], device=device).view(1, 1).long()
    state = None

    # 此循环主要是获得state数据，从而得到意境
    if prefix_words:
        for word in prefix_words:
            _, state = model(X, state)
            X = X.data.new([char_to_idx[word]]).view(1, 1)  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致

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
        if w == '<EOP>':
            del results[-1]
            break
    model.train()
    return results


@torch.no_grad()
def poem_val_acrostic_generate(model, start_words, idx_to_char, char_to_idx, device, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    model.eval()
    results = []
    start_word_len = len(start_words)

    # 手动设置第一个词为<START>
    X = torch.tensor([char_to_idx['<START>']], device=device).view(1, 1).long()
    state = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'

    # 此循环主要是获得state数据，从而得到意境
    if prefix_words:
        for word in prefix_words:
            _, state = model(X, state)
            X = X.data.new([char_to_idx[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        Y, state = model(X, state)

        top_index = Y.argmax(dim=1).long().item()
        w = idx_to_char[top_index]


        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                X = X.data.new([char_to_idx[w]]).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            X = X.data.new([char_to_idx[w]]).view(1, 1)
        results.append(w)
        pre_word = w
    return results






def predict_rnn_pytorch(prefix, num_chars, model, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, data_iter, criterion, optimizer, device,
                                  idx_to_char, char_to_idx,
                                  num_epochs, clipping_theta,
                                  pred_period, pred_len, prefixes):

    model.to(device)
    state = None
    for epoch in range(num_epochs):
        loss_sum, n, start = 0.0, 0, time.time()
        data_iter.reset()

        for X, Y in data_iter():  # X与Y的形状均为(batch_size, num_steps)
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            loss = criterion(output, y.long())

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            params = [p for p in model.parameters()]
            grad_clipping(params, clipping_theta, device)

            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(loss_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, device, idx_to_char,
                    char_to_idx))

