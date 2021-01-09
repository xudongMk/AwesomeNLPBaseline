# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 22:59
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : inference.py
# @Software: PyCharm

import tensorflow as tf
import tqdm
import json
import _pickle as cPickle

"""
命名实体识别推理代码
"""

# 加载词典
word_dict = {}
with open('./data_path/clue_vocab.txt', encoding='utf-8') as fr:
    lines = fr.readlines()
for line in lines:
    word = line.split('\t')[0]
    id = line.split('\t')[1]
    word_dict[word] = id
print(word_dict)

# label dict的设置 这个和preprocess中的tag_dict对应
tag_ids = {0: 'O', 1: 'B-address', 2: 'I-address', 3: 'B-book', 4: 'I-book',
           5: 'B-company', 6: 'I-company', 7: 'B-game', 8: 'I-game',
           9: 'B-government', 10: 'I-government', 11: 'B-movie', 12: 'I-movie',
           13: 'B-name', 14: 'I-name', 15: 'B-organization', 16: 'I-organization',
           17: 'B-position', 18: 'I-position', 19: 'B-scene', 20: 'I-scene'}


def words_to_ids(words, word_dict):
    """ 将words 转换成ids形式 """
    ids = [word_dict.get(word, 1) for word in words]
    return ids


def predict_main(test_file, out_path):
    """ 预测主入口 """
    model_path = './model_pb/1609946529'
    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(sess, ['serve'], model_path)
        # print(model)
        out = sess.graph.get_tensor_by_name('tag_ids:0')
        input_id = sess.graph.get_tensor_by_name('input_words:0')
        input_len = sess.graph.get_tensor_by_name('input_len:0')

        with open(test_file, encoding='utf-8') as fr:
            lines = fr.readlines()
        res_list = []

        cnt = 0
        for line in tqdm.tqdm(lines):
            json_str = json.loads(line)
            id = json_str['id']
            text = json_str['text']
            if len(text) < 1:
                print('there are some sample error!')
            text_features = words_to_ids(text, word_dict)
            text_label = len(text)
            feed = {input_id: [text_features], input_len: [text_label]}
            score = sess.run(out, feed_dict=feed)

            cnt += 1
            tags = [tag_ids[tag] for tag in score[0]]
            # print(tags)
            res_words, res_pos = get_result(text, tags)
            rs = {}
            for w, t in zip(res_words, res_pos):
                rs[t] = rs.get(t, []) + [w]
            pres = {}
            for t, ws in rs.items():
                temp = {}
                for w in ws:
                    word = text[w[0]: w[1] + 1]
                    temp[word] = temp.get(word, []) + [w]
                pres[t] = temp
            output_line = json.dumps({'id': id, 'label': pres}, ensure_ascii=False)
            res_list.append(output_line)
            # print(output_line)
        # write to file
        with open(out_path, 'w', encoding='utf-8') as fr:
            for res in res_list:
                fr.write(res)
                fr.write('\n')


def get_result(text, tags):
    """ 改写成clue要提交的格式 """
    result_words = []
    result_pos = []
    temp_word = []
    temp_pos = ''
    for i in range(min(len(text), len(tags))):
        if tags[i].startswith('O'):
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
        elif tags[i].startswith('B-'):
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = [i]
            temp_pos = tags[i].split('-')[1]
        elif tags[i].startswith('I-'):
            if len(temp_word) > 0:
                temp_word.append(i)
                if temp_pos == '':
                    temp_pos = tags[i].split('-')[1]
        else:
            if len(temp_word) > 0:
                temp_word.append(i)
                if temp_pos == '':
                    temp_pos = tags[i].split('-')[1]
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
    return result_words, result_pos


if __name__ == '__main__':
    test_file = './data_path/test.json'
    out_path = './data_path/clue_predict.json'
    predict_main(test_file, out_path)
