# -*- coding: utf-8 -*-
# @Time    : 2020-10-11 18:52
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : preprocess.py
# @Software: PyCharm

import os
import _pickle as pickle
import pandas as pd
import random

from sklearn.model_selection import train_test_split

"""
数据预处理
将数据处理成id，并封装成pkl形式
"""

# 可以人为自定义label dict
label_dict_default = {109: 0, 104: 1, 102: 2, 113: 3,
                      107: 4, 101: 5, 103: 6, 110: 7,
                      108: 8, 116: 9, 112: 10, 115: 11,
                      106: 12, 100: 13, 114: 14}


def make_vocab(file_path):
    """
    构建词典和label映射词典
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path, sep='\t')
    vocab = {'PAD': 0, 'UNK': 1}
    words_list = []
    for index, row in data.iterrows():
        label = row['label']
        words = row['words'].split(' ')
        for word in words:
            words_list.append(word)
    random.shuffle(words_list)
    for word in words_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    # save to file and print the label dict
    save_path = './data_path/vocab.txt'
    save_vocab(vocab, save_path)
    print(f'the vocab size is {len(vocab)}')
    return vocab


def make_data(file_path, vocab, type):
    """
    构建数据
    :param file_path:
    :param vocab
    :return:
    """
    data = pd.read_csv(file_path, sep='\t')
    word_ids = []
    label_ids = []
    for index, row in data.iterrows():
        label = row['label']
        words = row['words'].split(' ')
        word_id_temp = [vocab.get(word) if word in vocab else 1 for word in words]
        word_ids.append(word_id_temp)
        label_ids.append(label_dict_default.get(label))

    print(f'the {type} data size is {len(word_ids)}')
    print(word_ids[0])
    print(label_ids[0])

    return {'words': word_ids, 'labels': label_ids}


def save_vocab(vocab, output):
    """
    保存vocab到本地文件
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
    train_path = './data_path/train_data.csv'
    test_path = './data_path/dev_data.csv'
    vocab = make_vocab(train_path)
    train_data = make_data(train_path, vocab, 'train')
    test_data = make_data(test_path, vocab, 'test')

    data['train'] = train_data
    data['test'] = test_data

    data_path = os.path.join(output_path, 'tnews_data.pkl')
    pickle.dump(data, open(data_path, 'wb'), protocol=2)
    print('save data to pkl over.')


def split_data(file_path, output):
    """
    划分数据集
    :param file_path:
    :param output:
    :return:
    """
    all_data = pd.read_csv(file_path, sep='\t', header=None)
    all_data.columns = ['id', 'texta', 'textb', 'label']
    train_data, test_data = train_test_split(all_data, stratify=all_data['label'],
                                             test_size=0.2, shuffle=True,
                                             random_state=42)
    print(train_data)
    print(test_data)
    train_path = os.path.join(output, 'train_nli.csv')
    test_path = os.path.join(output, 'dev_nli.csv')
    train_data.to_csv(train_path, sep='\t', header=False, index=False)
    test_data.to_csv(test_path, sep='\t', header=False, index=False)
    print(f'split data train size={len(train_data)} test size={len(test_data)}')


if __name__ == '__main__':
    output_path = './data_path'
    main(output_path)
