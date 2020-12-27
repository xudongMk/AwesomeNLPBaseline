# -*- coding: utf-8 -*-
# @Time    : 2020-10-11 18:52
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : preprocess.py
# @Software: PyCharm

import os
import _pickle as cPickle
import pandas as pd
import random

"""
数据预处理
将数据处理成id，并封装成pkl形式
"""


# clue2020细粒度命名实体识别的类别
tag_list = ['address', 'book', 'company', 'game', 'government',
            'movie', 'name', 'organization', 'position', 'scene']
tag_dict = {'O': 0}

for tag in tag_list:
    tag_B = 'B-' + tag
    tag_I = 'I-' + tag
    tag_dict[tag_B] = len(tag_dict)
    tag_dict[tag_I] = len(tag_dict)

print(tag_dict)


def make_vocab(file_path):
    """
    构建词典
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ['text', 'tag']
    vocab = {'PAD': 0, 'UNK': 1}
    words_list = []
    for index, row in data.iterrows():
        words = row['text']
        for word in words:
            words_list.append(word)

    random.shuffle(words_list)
    for word in words_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def make_data(file_path, vocab):
    """
    构建数据
    :param file_path:
    :param vocab
    :return:
    """
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ['text', 'tag']
    word_ids = []
    tag_ids = []
    for index, row in data.iterrows():
        tag_str = row['tag']
        tags = tag_str.split(' ')
        words = row['text']

        word_id = [vocab.get(word) if word in vocab else 1 for word in words]
        tag_id = [tag_dict.get(tag) for tag in tags]

        word_ids.append(word_id)
        tag_ids.append(tag_id)
    print(word_ids[0])
    print(tag_ids[0])
    return {'words': word_ids, 'tags': tag_ids}


def save_vocab(vocab, output):
    """
    save vocab dict
    :param vocab:
    :param output:
    :return:
    """
    with open(output, 'w', encoding='utf-8') as fr:
        for word in vocab:
            fr.write(word + '\t' + str(vocab.get(word)) + '\n')
    print('save vocab is ok.')


def main(output_path):
    """
    main method
    :param output_path:
    :return:
    """
    data = {}
    # 这里是bio形式的数据集，如果不是需要提前转换成bio形式
    train_path = './data_path/train.txt'
    test_path = './data_path/dev.txt'
    vocab = make_vocab(train_path)
    train_data = make_data(train_path, vocab)
    test_data = make_data(test_path, vocab)

    data['train'] = train_data
    data['test'] = test_data

    data_path = os.path.join(output_path, 'clue_data.pkl')
    cPickle.dump(data, open(data_path, 'wb'), protocol=2)
    print('save data to pkl ok.')

    vocab_path = os.path.join(output_path, 'clue_vocab.txt')
    save_vocab(vocab, vocab_path)


if __name__ == '__main__':
    output = './data_path/'
    main(output)
