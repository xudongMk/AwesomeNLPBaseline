# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 20:34
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : bilstm_crf.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell


class Linear:
    """
    全链接层
    """
    def __init__(self, scope_name, input_size, output_size,
                 drop_out=0., trainable=True):
        with tf.variable_scope(scope_name):
            w_init = tf.random_uniform_initializer(-0.1, 0.1)
            self.W = tf.get_variable('W', [input_size, output_size],
                                     initializer=w_init,
                                     trainable=trainable)

            self.b = tf.get_variable('b', [output_size],
                                     initializer=tf.zeros_initializer(),
                                     trainable=trainable)

        self.drop_out = tf.layers.Dropout(drop_out)

        self.output_size = output_size

    def __call__(self, inputs, training):
        size = tf.shape(inputs)
        input_trans = tf.reshape(inputs, [-1, size[-1]])
        input_trans = tf.nn.xw_plus_b(input_trans, self.W, self.b)
        input_trans = self.drop_out(input_trans, training=training)

        input_trans = tf.reshape(input_trans, [-1, size[1], self.output_size])

        return input_trans


class LookupTable:
    """
    embedding layer
    """
    def __init__(self, scope_name, vocab_size, embed_size, reuse=False, trainable=True):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        with tf.variable_scope(scope_name, reuse=bool(reuse)):
            self.embedding = tf.get_variable('embedding', [vocab_size, embed_size],
                                             initializer=tf.random_uniform_initializer(-0.25, 0.25),
                                             trainable=trainable)

    def __call__(self, input):
        input = tf.where(tf.less(input, self.vocab_size), input, tf.ones_like(input))
        return tf.nn.embedding_lookup(self.embedding, input)


class LstmBase:
    """
    build rnn cell
    """
    def build_rnn(self, hidden_size, num_layes):
        cells = []
        for i in range(num_layes):
            cell = LSTMCell(num_units=hidden_size,
                            state_is_tuple=True,
                            initializer=tf.random_uniform_initializer(-0.25, 0.25))
            cells.append(cell)
        cells = MultiRNNCell(cells, state_is_tuple=True)

        return cells


class BiLstm(LstmBase):
    """
    define the lstm
    """
    def __init__(self, scope_name, hidden_size, num_layers):
        super(BiLstm, self).__init__()
        assert hidden_size % 2 == 0
        hidden_size /= 2

        self.fw_rnns = []
        self.bw_rnns = []
        for i in range(num_layers):
            self.fw_rnns.append(self.build_rnn(hidden_size, 1))
            self.bw_rnns.append(self.build_rnn(hidden_size, 1))

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


class BiLstm_Crf:
    def __init__(self, args, vocab_size, emb_size):
        # embedding
        scope_name = 'look_up'
        self.lookuptables = LookupTable(scope_name, vocab_size, emb_size)

        # rnn
        scope_name = 'bi_lstm'
        self.rnn = BiLstm(scope_name, args.hidden_dim, 1)

        # linear
        scope_name = 'linear'
        self.linear = Linear(scope_name, args.hidden_dim, args.num_tags,
                             drop_out=args.drop_out)

        # crf
        scope_name = 'crf_param'
        self.crf_param = tf.get_variable(scope_name, [args.num_tags, args.num_tags],
                                         dtype=tf.float32)

    def __call__(self, inputs, training):
        masks = tf.sign(inputs)
        sent_len = tf.reduce_sum(masks, axis=1)

        embedding = self.lookuptables(inputs)

        rnn_out = self.rnn(embedding, sent_len)

        logits = self.linear(rnn_out, training)

        pred_ids, _ = tf.contrib.crf.crf_decode(logits, self.crf_param, sent_len)

        return logits, pred_ids, self.crf_param




