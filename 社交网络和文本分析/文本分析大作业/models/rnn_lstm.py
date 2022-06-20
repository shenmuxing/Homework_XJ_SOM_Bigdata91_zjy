# coding:utf8
import torch
import torch.nn as nn


class Shenmuxing_RNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, bidirectional=False, *args, **kwargs):
        super(Shenmuxing_RNNLSTMModel, self).__init__()
        self.model_name = 'Shenmuxing_lstm'
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,num_layers=1,
                            bidirectional=bidirectional, *args, ** kwargs)
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dense = nn.Linear(self.hidden_dim, vocab_size)
        self.state = None

    def forward(self, input, state):
        # input size: (batch_size, seq_len)      input_T size: (seq_len, batch_size)
        input_T = input.transpose(0, 1).contiguous()

        # embeds size: (seq_len,batch_size, embeding_dim)
        embeds = self.embeddings(input_T)
        # Y size: (seq_len,batch_size,hidden_dim)
        Y, self.state = self.lstm(embeds, state)
        # output size: (seq_len*batch_size,vocab_size), 相当于对seq_len * batch_size个字做预测
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


class RNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, bidirectional=False, *args, **kwargs):
        super(RNNLSTMModel, self).__init__()
        self.model_name = 'lstm'
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            bidirectional=bidirectional, *args, ** kwargs)
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dense = nn.Linear(self.hidden_dim, vocab_size)
        self.state = None

    def forward(self, input, state):
        # input size: (batch_size, seq_len)      input_T size: (seq_len, batch_size)
        input_T = input.transpose(0, 1).contiguous()

        # embeds size: (seq_len,batch_size, embeding_dim)
        embeds = self.embeddings(input_T)
        # Y size: (seq_len,batch_size,hidden_dim)
        Y, self.state = self.lstm(embeds, state)
        # output size: (seq_len*batch_size,vocab_size), 相当于对seq_len * batch_size个字做预测
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

