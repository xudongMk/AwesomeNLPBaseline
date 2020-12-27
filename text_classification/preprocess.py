# -*- coding: utf-8 -*-
# @Time    : 2020-10-11 18:52
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : preprocess.py
# @Software: PyCharm

import os
import _pickle as pickle
import pandas as pd
import jieba
import random

from sklearn.model_selection import train_test_split

"""
数据预处理
将数据处理成id，并封装成pkl形式
"""

# 可以人为自定义label dict
label_dict_default = {108: 0, 104: 1, 106: 2, 112: 3,
                      109: 4, 103: 5, 116: 6, 101: 7,
                      107: 8, 100: 9, 102: 10, 110: 11,
                      115: 12, 113: 13, 114: 14}

emotion_dict_default = {'sadness': 0, 'anger': 1,
                        'happiness': 2, 'fear': 3,
                        'like': 4, 'disgust': 5, 'surprise': 6}


def make_vocab(file_path):
    """
    构建词典和label映射词典
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ['id', 'text', 'label']
    vocab = {'PAD': 0, 'UNK': 1}
    words_list = []
    label_dict = {}
    for index, row in data.iterrows():
        # todo 也可以使用分词软件进行分词
        text = jieba.lcut(row['text'])
        label = row['label']
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        for word in text:
            words_list.append(word)
    random.shuffle(words_list)
    for word in words_list:
        if word not in vocab:
            vocab[word] = len(vocab)
    # save to file and print the label dict
    save_path = './vocab.txt'
    save_vocab(vocab, save_path)
    print(vocab)
    print(f'the vocab size is {len(vocab)}')
    print(f'the label dict is : {label_dict}')
    return vocab


def make_data(file_path, vocab, type):
    """
    构建数据
    :param file_path:
    :param vocab
    :return:
    """
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ['id', 'text', 'label']
    word_ids = []
    label_ids = []
    for index, row in data.iterrows():
        # todo 也可以使用分词软件进行分词，然后基于词粒度来做
        text = jieba.lcut(row['text'])
        label = row['label']
        word_id_temp = [vocab.get(word) if word in vocab else 1 for word in text]
        word_ids.append(word_id_temp)
        label_ids.append(emotion_dict_default.get(label))

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
    train_path = './data_path/train_emotion.csv'
    test_path = './data_path/test_motion.csv'
    vocab = make_vocab(train_path)
    train_data = make_data(train_path, vocab, 'train')
    test_data = make_data(test_path, vocab, 'test')

    data['train'] = train_data
    data['test'] = test_data

    data_path = os.path.join(output_path, 'emotion_data.pkl')
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
    # main(output_path)
    # todo filter 一些特殊字符，and 停用词表
    split_data('./data_path/OCNLI_train1128.csv', './data_path/')
