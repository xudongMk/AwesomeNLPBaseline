# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 21:51
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : attention.py
# @Software: PyCharm

import tensorflow as tf
from .base_model import Linear


class Attention:
    """
    the attention
    """
    def __init__(self, scope_name, hidden_size, num_heads, dropout):
        if hidden_size % num_heads != 0:
            raise ValueError('the hidden size and heads is not match!')

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q_layer = Linear(f'{scope_name}_q', hidden_size, hidden_size, bias=False)
        self.k_layer = Linear(f'{scope_name}_k', hidden_size, hidden_size, bias=False)
        self.v_layer = Linear(f'{scope_name}_v', hidden_size, hidden_size, bias=False)

        self.out_layer = Linear(f'{scope_name}_output', hidden_size,
                                hidden_size, bias=False)
        self.dropout = tf.layers.Dropout(dropout)

    def split_heads(self, x):
        """ split the heads """
        with tf.name_scope('split_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            depth = self.hidden_size // self.num_heads

            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """ combine the heads """
        with tf.name_scope('combine_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]

            x = tf.transpose(x, [0, 2, 1, 3])  # bacth length, heads, depth
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, training, bias, cache=None):
        q = self.q_layer(x, training)
        k = self.k_layer(y, training)
        v = self.v_layer(y, training)

        if cache:
            k = tf.concat([cache['k'], k], axis=1)
            v = tf.concat([cache['v'], v], axis=1)

            cache['k'] = k
            cache['v'] = v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = self.hidden_size // self.num_heads
        q *= depth ** -0.5

        # calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits)
        weights = self.dropout(weights, training=training)
        attention_output = tf.matmul(weights, v)

        attention_output = self.combine_heads(attention_output)

        attention_output = self.out_layer(attention_output, training)
        return attention_output


class SelfAttention(Attention):
    def __call__(self, x, training, bias, cache=None):
        return super(SelfAttention, self).call(x, x, training, bias, cache)