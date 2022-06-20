import warnings
import torch as t


class Config(object):
    jieba_rate=0.1 #MArx模型使用jieba分词的比率
    data_path = 'data/poem/'  # 诗歌的文本文件存放路径
    pickle_path = 'data/poem/tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    iter_consecutive = False   # batch与batch之间的文本是否有连贯性，若dataloader中shuffle参数为True，则该参数一定为False
    max_epoch = 20
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_decay_step = 20  # Period of learning rate decay (每隔lr_decay_step个epoch学习率缩小lr_decay倍)
    clipping_grad = False
    clipping_theta = 1e-2  # 对于梯度的裁剪阈值，在simple rnn中使用
    batch_size = 64
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面padding
    print_freq = 200  # 每20个batch 可视化一次
    save_freq = 10  # 每10个epoch保存一次训练模型
    max_gen_len = 200  # 生成诗歌最长长度
    model_path = None  # 预训练模型路径
    prefix_words = u'江流天地外，山色有无中。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = u'秦時明月漢時關'  # 诗歌开始
    padding = 'pre' # 若单个训练数据样本的字数没达到maxlen字数，从前补充还是从后补充（默认pre，在前面补充）
    truncating = 'post'  # 若单个训练数据样本的字数超过maxlen字数，从前裁剪还是从后裁剪（默认post，从后裁剪）
    acrostic = True  # 是否是藏头诗
    criterion = t.nn.CrossEntropyLoss()


    if t.cuda.is_available() & use_gpu:
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))



opt = Config()