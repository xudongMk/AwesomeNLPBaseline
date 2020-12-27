# -*- coding: utf-8 -*-
# @Time    : 2020-12-11 19:05
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : ffnn_model.py
# @Software: PyCharm

import tensorflow as tf

from .base_model import LookupTable
from .base_model import Linear


class FCModel:
    """
    前馈网络
    主要包括：embedding层、两个全连接层和一个dropout层
    """
    def __init__(self, vocab_size, emb_size, args):

        # embedding
        scope_name = 'look_up'
        self.lookuptables = LookupTable(scope_name, vocab_size, emb_size)

        fc_layer_size = args.fc_layer_size  # 全链接层的size
        scope_name = 'linear1'
        self.linear1 = Linear(scope_name, emb_size, fc_layer_size, args.activator)

        scope_name = 'linear2'
        self.linear2 = Linear(scope_name, fc_layer_size, args.num_label)

        self.dropout = tf.layers.Dropout(args.drop_out)

    def __call__(self, inputs, training):
        embedding = self.lookuptables(inputs)
        pool_out = tf.reduce_mean(embedding, 1)
        pool_out = self.dropout(pool_out, training)
        pool_out = self.linear1(pool_out, training)
        pool_out = self.dropout(pool_out, training)
        output = self.linear2(pool_out, training)

        return output
