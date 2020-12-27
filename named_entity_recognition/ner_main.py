# -*- coding: utf-8 -*-
# @Time    : 2020-10-09 23:07
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : ner_main.py
# @Software: PyCharm

import sys
import time
import tensorflow as tf
from data_utils import datasets

import _pickle as cPickle

from argparse import ArgumentParser
from models.bilstm_crf import BiLstm_Crf

parser = ArgumentParser()

parser.add_argument("--vocab_size", type=int, default=4000, help='vocab size')
parser.add_argument("--emb_size", type=int, default=300, help='emb size')
parser.add_argument("--train_path", type=str, default='./data_path/clue_data.pkl')
parser.add_argument("--test_path", type=str, default='./data_path/clue_data.pkl')
parser.add_argument("--model_ckpt_dir", type=str, default='./model_ckpt/')
parser.add_argument("--model_pb_dir", type=str, default='./model_pb')
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--num_tags", type=int, default=21)
parser.add_argument("--drop_out", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--type", type=str, default='lstm', help='[lstm, textcnn...]')
parser.add_argument("--lr", type=float, default=1e-4,
                    help='the learning rate for optimizer')


tf.logging.set_verbosity(tf.logging.INFO)
ARGS, unparsed = parser.parse_known_args()
print(ARGS)

sys.stdout.flush()


def init_data(file_name, type=None):
    """
    init data
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
    build model
    :return:
    """
    vocab_size = ARGS.vocab_size
    emb_size = ARGS.emb_size

    if ARGS.type == 'lstm':
        model = BiLstm_Crf(ARGS, vocab_size, emb_size)
    else:
        pass

    return model


def model_fn(features, labels, mode, params):
    """
    build model fn
    :return:
    """
    model = make_model()

    if isinstance(features, dict):
        features = features['words'], features['words_len']

    words, words_len = features

    if mode == tf.estimator.ModeKeys.PREDICT:
        _, pred_ids, _ = model(words, training=False)

        prediction = {'tag_ids': tf.identity(pred_ids, name='tag_ids')}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction,
            export_outputs={'classify': tf.estimator.export.PredictOutput(prediction)}
        )
    else:
        tags = labels
        weights = tf.sequence_mask(words_len)
        if mode == tf.estimator.ModeKeys.TRAIN:
            logits, pred_ids, crf_params = model(words, training=True)

            log_like_lihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, words_len, crf_params
            )
            loss = -tf.reduce_mean(log_like_lihood)
            accuracy = tf.metrics.accuracy(tags, pred_ids, weights)

            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
            )
        else:
            _, pred_ids, _ = model(words, training=False)
            accuracy = tf.metrics.accuracy(tags, pred_ids, weights)
            metrics = {
                'accuracy': accuracy
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

    # train
    train_input = init_data(ARGS.train_path, 'train')
    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    classifer.train(input_fn=train_input, hooks=[logging_hook])

    # eval
    test_input = init_data(ARGS.test_path, 'test')
    eval_res = classifer.evaluate(input_fn=test_input)
    print(f'Evaluation res is : \n\t{eval_res}')

    if ARGS.model_pb_dir:
        words = tf.placeholder(tf.int64, [None, None], name='input_words')
        words_len = tf.placeholder(tf.int64, [None], name='input_len')
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'words': words,
            'words_len': words_len
        })
        classifer.export_savedmodel(ARGS.model_pb_dir, input_fn)


if __name__ == '__main__':
    tf.app.run(main=main_es, argv=[sys.argv[0]])