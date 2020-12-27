# -*- coding: utf-8 -*-
# @Time    : 2020-12-11 19:47
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : train_main.py
# @Software: PyCharm

import sys
import os
import time
import tensorflow as tf
import tf_metrics
import _pickle as cPickle

from data_utils import datasets
from argparse import ArgumentParser
from models.bilstm_model import BiLstmModel
from models.text_cnn import TextCnn
from models.ffnn_model import FCModel


# 设置参数
parser = ArgumentParser()

parser.add_argument("--train_path", type=str, default='./data_path/imdb.pkl',
                    help='the file path of train data, needs pkl type')
parser.add_argument("--test_path", type=str, default='./data_path/imdb.pkl',
                    help='the file path of test data, needs pkl type')
parser.add_argument("--model_ckpt_dir", type=str, default='./model_ckpt/',
                    help='the dir of the checkpoint type model')
parser.add_argument("--model_pb_dir", type=str, default='./model_pb',
                    help='the dir of the pb type model')

parser.add_argument("--vocab_size", type=int, default=20000, help='the vocab size')
parser.add_argument("--emb_size", type=int, default=300, help='the embedding size')
parser.add_argument("--hidden_size", type=int, default=300,
                    help='the hidden size of rnn layer, will split it half in rnn')
parser.add_argument("--fc_layer_size", type=int, default=300,
                    help='the hidden size of fully connect layer')
parser.add_argument("--num_label", type=int, default=2, help='the number of task label')
parser.add_argument("--drop_out", type=float, default=0.2,
                    help='the dropout rate in layers')
parser.add_argument("--batch_size", type=int, default=16,
                    help='the batch size of dataset in one step training')
parser.add_argument("--epoch", type=int, default=5,
                    help='the epoch count we want to train')
parser.add_argument("--model_name", type=str, default='lstm',
                    help='which model we want use in our task, [lstm, cnn, fc, ...]')
parser.add_argument("--pool", type=str, default='max',
                    help='the pool function, [max, mean, ...]')
parser.add_argument("--activator", type=str, default='relu',
                    help='the activate function, [relu, relu6, tanh, ...]')
parser.add_argument("--filter_num", type=int, default=128,
                    help='the number of the cnn filters')
parser.add_argument("--use_pos", type=int, default=0,
                    help='whether to use position encoding in embedding layer')
parser.add_argument("--lr", type=float, default=1e-4,
                    help='the learning rate for optimizer')


# todo 还可以加入位置信息在embedding层
# todo pool层还可以加入attention pool


tf.logging.set_verbosity(tf.logging.INFO)
ARGS, unparsed = parser.parse_known_args()
print(ARGS)
sys.stdout.flush()


def init_data(file_name, type=None):
    """
    初始化数据集并构建input function
    :param file_name:
    :param type:
    :return:
    """
    data = cPickle.load(open(file_name, 'rb'))[type]

    data_builder = datasets.DataBuilder(data)
    dataset = data_builder.build_dataset()

    def train_input():
        return data_builder.get_train_batch(dataset, ARGS.batch_size, ARGS.epoch)

    def test_input():
        return data_builder.get_test_batch(dataset, ARGS.batch_size)

    return train_input if type == 'train' else test_input


def make_model():
    """
    构建模型
    :return:
    """
    vocab_size = ARGS.vocab_size
    emb_size = ARGS.emb_size
    print(f'the model name is {ARGS.model_name}')
    if ARGS.model_name == 'lstm':
        model = BiLstmModel(vocab_size, emb_size, ARGS)
    elif ARGS.model_name == 'cnn':
        model = TextCnn(vocab_size, emb_size, ARGS)
    elif ARGS.model_name == 'fc':
        model = FCModel(vocab_size, emb_size, ARGS)
    else:
        raise KeyError('the model type is not implemented!')
    return model


def model_fn(features, labels, mode, params):
    """
    the model fn
    :return:
    """
    model = make_model()

    if isinstance(features, dict):
        features = features['words']

    words = features

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(words, training=False)

        prediction = {'class_id': tf.argmax(logits, axis=1, name='class_out'),
                      'prob': tf.nn.softmax(logits, name='prob_out')}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction,
            export_outputs={'classify': tf.estimator.export.PredictOutput(prediction)}
        )
    else:
        if mode == tf.estimator.ModeKeys.TRAIN:
            logits = model(words, training=True)
            # weights = tf.constant([])
            # weights = tf.gather(weights, labels)
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels, logits,
                # weights=weights,
                reduction=tf.losses.Reduction.MEAN)
            prediction = tf.argmax(logits, axis=1)
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=prediction)
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])
            optimizer = tf.train.AdamOptimizer(learning_rate=ARGS.lr)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
            )
        else:
            logits = model(words, training=False)
            prediction = tf.argmax(logits, axis=1)
            precision = tf_metrics.precision(labels, prediction, ARGS.num_label)
            recall = tf_metrics.recall(labels, prediction, ARGS.num_label)
            accuracy = tf.metrics.accuracy(labels, predictions=prediction)
            metrics = {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.constant(0),
                eval_metric_ops=metrics
            )


def main_es(unparsed):
    """
    main method
    :param unparsed:
    :return:
    """
    cur_time = time.time()
    model_dir = ARGS.model_ckpt_dir + str(int(cur_time))

    classifer = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={}
    )

    # train model
    train_input = init_data(ARGS.train_path, 'train')
    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    classifer.train(input_fn=train_input, hooks=[logging_hook])

    # eval model
    test_input = init_data(ARGS.test_path, 'test')
    eval_res = classifer.evaluate(input_fn=test_input)
    print(f'Evaluation res is : \n\t{eval_res}')

    if ARGS.model_pb_dir:
        words = tf.placeholder(tf.int64, [None, None], name='input_words')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'words': words
        })
        classifer.export_savedmodel(ARGS.model_pb_dir, input_fn)


if __name__ == '__main__':
    tf.app.run(main=main_es, argv=[sys.argv[0]] + unparsed)
