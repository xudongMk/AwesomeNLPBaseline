# -*- coding: utf-8 -*-
# @Time    : 2020/12/27 13:02
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : predict_multi_learning.py
# @Software: PyCharm

import tensorflow as tf
import pandas as pd
import json

from bert_master import tokenization


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(text, max_seq_length, tokenizer, task):
    """Converts a single text to `InputFeatures`."""
    text_s = text.split('\t')
    text_id = text_s[0]
    text_a = text_s[1]
    text_b = None
    if task == 3:  # 3=nli
        text_b = text_s[2]

    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = (text_id, input_ids, input_mask, segment_ids)
    return feature


def load_model(model_dir):
    """ 加载模型 """
    try:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("the input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_dir
        print("the Model folder", model_dir, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    sess_ = tf.Session()
    saver.restore(sess_, input_checkpoint)

    # opts = sess_.graph.get_operations()
    # for v in opts:
    #     print(v.name)
    return sess_


# 设置模型图的tensor
model_dir = './model_ckpt/multi_learning/'
sess = load_model(model_dir)
input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")
segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")
task = sess.graph.get_tensor_by_name("task:0")
pre_id = sess.graph.get_tensor_by_name("pre_id:0")


def predict_batch(text, tokenizer, max_seq_len, task):
    """ 一个batch的预测 """
    if isinstance(text, list):
        data = text
    else:
        data = [text]
    features = []
    text_ids = []
    for i_text in data:
        feature = convert_single_example(i_text, max_seq_len, tokenizer, task)
        features.append(feature)
        text_ids.append(feature[0])
    print([feature[1] for feature in features])
    feed = {input_ids: [feature[1] for feature in features],
            input_mask: [feature[2] for feature in features],
            segment_ids: [feature[3] for feature in features],
            task: task}

    pre_id_res = sess.run([pre_id], feed_dict=feed)
    return text_ids, pre_id_res


# 设置参数
batch_size = 8
max_seq_len = 128
vocab_file = './pre_trained/vocab.txt'
tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

label_list = [['sadness', 'anger', 'happiness', 'fear', 'like',
               'disgust', 'surprise'],
              ['108', '104', '106', '112', '109', '103', '116', '101',
               '107', '100', '102', '110', '115', '113', '114'],
              ['0', '1', '2']]

emotion_label_map = {}
news_label_map = {}
nli_label_map = {}
for (i, label) in enumerate(label_list[0]):
    emotion_label_map[i] = label
for (i, label) in enumerate(label_list[1]):
    news_label_map[i] = label
for (i, label) in enumerate(label_list[2]):
    nli_label_map[i] = label


def main():
    emotion_dir = './data_path/test_emotion.csv'
    emotion_data = pd.read_csv(emotion_dir, sep='\t', encoding='utf-8')
    emotion_data.columns = ['id', 'text']
    emotion_res = []
    with open(emotion_dir, encoding='utf-8') as fr:
        lines = fr.readlines()
    for index, text in enumerate(lines):
        ids, pres = predict_batch(text, tokenizer, max_seq_len, 1)

        print(ids)
        print(pres)
        # emotion_dict
        #label = emotion_label_map.get(pre)
        #emotion_res.append(json.dumps({'id': str(id), 'label': str(label)}))
        break


if __name__ == '__main__':
    main()




