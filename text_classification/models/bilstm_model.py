# -*- coding: utf-8 -*-
# @Time    : 2020-12-11 12:37
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : bilstm_model.py
# @Software: PyCharm
import tensorflow as tf

from .base_model import LookupTable
from .base_model import BiLstm
from .base_model import Linear


class BiLstmModel:
    """
    BiLstm模型的实现：
    主要包含：embedding层、rnn层、池化层、两层全连接层和一个Dropout层
    """
    def __init__(self, vocab_size, emb_size, args):

        # embedding
        scope_name = 'look_up'
        self.lookuptables = LookupTable(scope_name, vocab_size, emb_size)

        # rnn
        scope_name = 'bi_lstm'
        # rnn的层数 这里设置为1
        num_layers = 1
        self.rnn = BiLstm(scope_name, args.hidden_size, num_layers)

        # linear1
        scope_name = 'linear1'
        self.linear1 = Linear(scope_name, args.hidden_size, args.fc_layer_size,
                              activator=args.activator)

        # logits out
        scope_name = 'linear2'
        self.linear2 = Linear(scope_name, args.fc_layer_size, args.num_label)

        self.dropout = tf.layers.Dropout(args.drop_out)

        def max_pool(inputs):
            return tf.reduce_max(inputs, 1)

        def mean_pool(inputs):
            return tf.reduce_mean(inputs, 1)

        if args.pool == 'max':
            self.pool = max_pool
        else:
            self.pool = mean_pool

    def __call__(self, inputs, training):
        masks = tf.sign(inputs)
        sent_len = tf.reduce_sum(masks, axis=1)

        embedding = self.lookuptables(inputs)

        rnn_out = self.rnn(embedding, sent_len)
        pool_out = self.pool(rnn_out)
        linear_out = self.linear1(pool_out, training)
        # dropout
        linear_out = self.dropout(linear_out, training)
        # linear
        output = self.linear2(linear_out, training)
        return output