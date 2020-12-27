# -*- coding: utf-8 -*-
# @Time    : 2020-12-11 19:19
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : text_cnn.py
# @Software: PyCharm

import tensorflow as tf

from .base_model import LookupTable
from .base_model import BiLstm
from .base_model import Linear


class TextCnn:
    """
    text cnn model
    主要包括：embedding层、三个不同size卷积核层、两个全连接层和dropout层
    """
    def __init__(self, vocab_size, emb_size, args):

        # embedding
        scope_name = 'look_up'
        self.lookuptables = LookupTable(scope_name, vocab_size, emb_size)

        # 三个卷积核
        kws = [2, 3, 5]
        self.conv_ws = []
        self.conv_bs = []

        # the num of filter 卷积核的数量
        filter_num = args.filter_num
        for idx, kw in enumerate(kws):
            w = tf.get_variable(
                f"conv_w_{idx}",
                [kw, emb_size, filter_num],
                initializer=tf.random_uniform_initializer(-0.25, 0.25)
            )
            b = tf.get_variable(
                f"conv_b_{idx}",
                [filter_num],
                initializer=tf.random_uniform_initializer(-0.25, 0.25)
            )
            self.conv_ws.append(w)
            self.conv_bs.append(b)

        scope_name = 'linear1'
        self.linear1 = Linear(scope_name, len(kws) * filter_num,
                              args.fc_layer_size, activator=args.activator)

        scope_name = 'linear2'
        self.linear2 = Linear(scope_name, args.fc_layer_size, args.num_label)

        self.dropout = tf.layers.Dropout(args.drop_out)

    def __call__(self, inputs, training):
        embedding = self.lookuptables(inputs)

        outputs = []
        for conv_w, conv_b in zip(self.conv_ws, self.conv_bs):
            conv = tf.nn.conv1d(embedding, conv_w, 1, 'SAME')
            conv = tf.nn.bias_add(conv, conv_b)
            pool = tf.reduce_max(conv, axis=1)
            outputs.append(pool)
        output = tf.concat(outputs, -1)
        output = self.linear1(output, training)
        output = self.dropout(output, training)
        output = self.linear2(output, training)
        return output