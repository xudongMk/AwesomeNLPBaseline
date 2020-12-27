# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 21:34
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : base_model.py
# @Software: PyCharm

import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import BasicRNNCell


class Linear:
    """
    线性层，全连接层
    """
    def __init__(self, scope_name, input_size, output_sizes, bias=True,
                 activator='', drop_out=0., reuse=False, trainable=True):
        self.input_size = input_size

        # todo 判断 output_sizes 是不是列表
        if not isinstance(output_sizes, list):
            output_sizes = [output_sizes]

        self.output_size = output_sizes[-1]

        self.W = []
        self.b = []
        size = input_size
        with tf.variable_scope(scope_name, reuse=reuse):
            for i, output_size in enumerate(output_sizes):
                W = tf.get_variable(
                    'W{0}'.format(i), [size, output_size],
                    initializer=tf.random_uniform_initializer(-0.25, 0.25),
                    trainable=trainable
                )
                if bias:
                    b = tf.get_variable(
                        'b{0}'.format(i), [output_size],
                        initializer=tf.zeros_initializer(),
                        trainable=trainable
                    )
                else:
                    b = None

                self.W.append(W)
                self.b.append(b)
                size = output_size

        if activator == 'relu':
            self.activator = tf.nn.relu
        elif activator == 'relu6':
            self.activator = tf.nn.relu6
        elif activator == 'tanh':
            self.activator = tf.nn.tanh
        else:
            self.activator = tf.identity

        self.drop_out = tf.layers.Dropout(drop_out)

    def __call__(self, input, training):
        size = tf.shape(input)
        input_trans = tf.reshape(input, [-1, size[-1]])
        for W, b in zip(self.W, self.b):
            if b is not None:
                input_trans = tf.nn.xw_plus_b(input_trans, W, b)
            else:
                input_trans = tf.matmul(input_trans, W)

            input_trans = self.drop_out(input_trans, training)
            input_trans = self.activator(input_trans)

        new_size = tf.concat([size[:-1], tf.constant([self.output_size])], 0)
        input_trans = tf.reshape(input_trans, new_size)
        return input_trans


class LookupTable:
    """
    embedding层
    """
    def __init__(self, scope_name, vocab_size, embed_size, reuse=False, trainable=True):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        with tf.variable_scope(scope_name, reuse=bool(reuse)):
            self.embedding = tf.get_variable(
                'embedding', [vocab_size, embed_size],
                initializer=tf.random_uniform_initializer(-0.25, 0.25),
                trainable=trainable
            )

    def __call__(self, input):
        input = tf.where(tf.less(input, self.vocab_size), input, tf.ones_like(input))
        return tf.nn.embedding_lookup(self.embedding, input)


class AttentionPooling:
    """
    attention pooling层
    """
    def __init__(self, scope_name, input_size, hidden_size, reuse=False,
                 trainable=True):
        name = scope_name
        self.linear1 = Linear(f'{name}_linear1', input_size,
                              hidden_size, bias=False, reuse=reuse,
                              trainable=trainable)
        self.linear2 = Linear(f'{name}_linear2', hidden_size, 1,
                              bias=False, reuse=reuse, trainable=trainable)

    def __call__(self, input, mask, training):
        output_linear1 = self.linear1(input, training)
        output_linear2 = self.linear2(output_linear1, training)
        weights = tf.squeeze(output_linear2, [-1])
        if mask is not None:
            weights += mask
        weights = tf.nn.softmax(weights, -1)
        return tf.reduce_sum(input * tf.expand_dims(weights, -1), axis=1)


class LayerNormalization:
    """
    归一化层
    """
    def __init__(self, scope_name, hidden_size):
        with tf.variable_scope(scope_name):
            self.scale = tf.get_variable('layer_norm_scale', [hidden_size],
                                         initializer=tf.ones_initializer())
            self.bias = tf.get_variable('layer_norm_bias', [hidden_size],
                                        initializer=tf.zeros_initializer())

    def __call__(self, x, epsilon=1e-6):
        mean, variance = tf.nn.moments(x, -1, keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class LstmBase:
    """
    RNN的基础层
    """
    def build_rnn(self, rnn_type, hidden_size, num_layes):
        cells = []
        for i in range(num_layes):
            if rnn_type == 'lstm':
                cell = LSTMCell(num_units=hidden_size,
                                state_is_tuple=True,
                                initializer=tf.random_uniform_initializer(-0.25, 0.25))
            elif rnn_type == 'gru':
                cell = GRUCell(num_units=hidden_size)
            elif rnn_type:
                cell = BasicRNNCell(num_units=hidden_size)
            else:
                raise NotImplementedError(f'the rnn type is unexist: {rnn_type}')
            cells.append(cell)

        cells = MultiRNNCell(cells, state_is_tuple=True)

        return cells


class BiLstm(LstmBase):
    """
    双向LSTM层
    """
    def __init__(self, scope_name, hidden_size, num_layers):
        super(BiLstm, self).__init__()
        assert hidden_size % 2 == 0
        hidden_size /= 2

        self.fw_rnns = []
        self.bw_rnns = []
        for i in range(num_layers):
            self.fw_rnns.append(self.build_rnn('lstm', hidden_size, 1))
            self.bw_rnns.append(self.build_rnn('lstm', hidden_size, 1))

        self.scope_name = scope_name

    def __call__(self, input, input_len):
        for idx, (fw_rnn, bw_rnn) in enumerate(zip(self.fw_rnns, self.bw_rnns)):
            scope_name = '{}_{}'.format(self.scope_name, idx)
            ctx, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_rnn, bw_rnn, input, sequence_length=input_len,
                dtype=tf.float32, time_major=False,
                scope=scope_name
            )
            input = tf.concat(ctx, -1)
        ctx = input
        return ctx


class Cnn:
    """
    define cnn
    """
    def __init__(self, scoep_name, input_size, hidden_size):
        kws=[3]
        self.conv_ws = []
        self.conv_bs = []
        for idx, kw in enumerate(kws):
            w = tf.get_variable(
                f"conv_w_{idx}",
                [kw, input_size, hidden_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1)
            )
            b = tf.get_variable(
                f"conv_b_{idx}",
                [hidden_size],
                initializer=tf.zeros_initializer()
            )
            self.conv_ws.append(w)
            self.conv_bs.append(b)

    def __call__(self, input, mask):
        outputs = []
        for conv_w, conv_b in zip(self.conv_ws, self.conv_bs):
            conv = tf.nn.conv1d(input, conv_w, 1, 'SAME')
            conv = tf.nn.bias_add(conv, conv_b)
            if mask is not None:
                conv += tf.expand_dims(mask, -1)
            outputs.append(conv)
        output = tf.concat(outputs, -1)
        return output
