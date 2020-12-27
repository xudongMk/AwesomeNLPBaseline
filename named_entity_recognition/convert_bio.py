# -*- coding: utf-8 -*-
# @Time    : 2020/12/15 21:46
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : convert_bio.py
# @Software: PyCharm
import json

"""
将数据转换成bio形式
"""


def read_data(file_path):
    """
    读取数据集
    :param file_path:
    :return:
    """
    with open(file_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    print(f'the data size is {len(lines)}')
    return lines


def convert_bio_data(file_path, out):
    """
    转换成bio形式
    example：
    我要去故宫   O O O B-location I-location
    :param file_path:
    :return:
    """
    lines = read_data(file_path)
    bio_data = []
    for line in lines:
        data = json.loads(line)
        text = data['text']
        labels = data['label']
        # 遍历处理label
        bios = ['O'] * len(text)
        for label in labels:
            entitys = labels[label]
            for entity in entitys:
                indexs = entitys[entity]
                for index in indexs:
                    start = index[0]
                    end = index[1]
                    for i in range(start, end + 1):
                        if i == start:
                            bios[i] = f'B-{label}'
                        else:
                            bios[i] = f'I-{label}'
        bio_data.append(text + '\t' + ' '.join(bios))
    # write to file
    with open(out, 'w', encoding='utf-8') as fr:
        for data in bio_data:
            fr.write(data + '\n')
    print(f'convert bio data over!')


if __name__ == '__main__':
    convert_bio_data('./data_path/train.json', './data_path/train.txt')
    convert_bio_data('./data_path/dev.json', './data_path/dev.txt')
