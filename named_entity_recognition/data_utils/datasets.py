# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 20:34
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : datasets.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf

"""
数据集构建类
将数据转换成模型所需要的dataset输入
"""


class DataBuilder:
    def __init__(self, data):
        self.words = np.asarray(data['words'])
        self.tags = np.asarray(data['tags'])

    @property
    def size(self):
        return len(self.words)

    def build_generator(self):
        """
        build data generator for model
        :return:
        """
        for word, tag in zip(self.words, self.tags):
            yield (word, len(word)), tag

    def build_dataset(self):
        """
        build dataset from generator
        :return:
        """
        dataset = tf.data.Dataset.from_generator(
            self.build_generator,
            ((tf.int64, tf.int64), tf.int64),
            ((tf.TensorShape([None]), tf.TensorShape([])), tf.TensorShape([None]))
        )
        return dataset

    def get_train_batch(self, dataset, batch_size, epoch):
        """
        get one batch train data
        :param dataset:
        :param batch_size:
        :param epoch:
        :return:
        """
        dataset = dataset.cache()\
            .shuffle(buffer_size=10000)\
            .padded_batch(batch_size, padded_shapes=(([None], []), [None]))\
            .repeat(epoch)
        return dataset.make_one_shot_iterator().get_next()

    def get_test_batch(self, dataset, batch_size):
        """
        get one batch test data
        :param dataset:
        :param batch_size:
        :return:
        """
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=(([None], []), [None]))
        return dataset.make_one_shot_iterator().get_next()
