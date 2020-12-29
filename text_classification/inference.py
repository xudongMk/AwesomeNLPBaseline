# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 21:45
# @Author  : xudong
# @email   : dongxu222mk@163.com
# @File    : inference.py
# @Software: PyCharm

import tensorflow as tf
import tqdm
import json
import jieba

"""
文本分类推理代码
"""

# 设置filter
filter = './?？；。（()）【】{}[]!！，,<>《》+'
# 加载词典
word_dict = {}
with open('./data_path/vocab.txt', encoding='utf-8') as fr:
    lines = fr.readlines()
for line in lines:
    word = line.split('\t')[0]
    id = line.split('\t')[1]
    word_dict[word] = id
print(word_dict)

# label dict的设置
label_id = {0: 109, 1: 104, 2: 102, 3: 113,
            4: 107, 5: 101, 6: 103, 7: 110,
            8: 108, 9: 116, 10: 112, 11: 115,
            12: 106, 13: 100, 14: 114}
label_desc = {100: "news_story", 101: "news_culture", 102: "news_entertainment",
              103: "news_sports", 104: "news_finance", 106: "news_house",
              107: "news_car", 108: "news_edu", 109: "news_tech",
              110: "news_military", 112: "news_travel", 113: "news_world",
              114: "news_stock", 115: "news_agriculture", 116: "news_game"}


def cut_with_jieba(text, filter=None):
    """ 使用jieba切分句子 """
    if filter:
        for c in filter:
            text = text.replace(c, '')
    words = ['Number' if word.isdigit() else word for word in jieba.cut(text)]
    return words


def words_to_ids(words, word_dict):
    """ 将words 转换成ids形式 """
    ids = [word_dict.get(word, 1) for word in words]
    return ids


def predict_main(test_file, out_path):
    """ 预测主入口 """
    model_path = './model_pb/1609247078'
    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(sess, ['serve'], model_path)
        # print(model)
        out = sess.graph.get_tensor_by_name('class_out:0')
        input_p = sess.graph.get_tensor_by_name('input_words:0')

        with open(test_file, encoding='utf-8') as fr:
            lines = fr.readlines()
        res_list = []
        for line in tqdm.tqdm(lines):
            json_str = json.loads(line)
            id = json_str['id']
            sentence = json_str['sentence']

            words = cut_with_jieba(str(sentence), filter)
            if len(words) < 1:
                print('there are some sample error!')
            text_features = words_to_ids(words, word_dict)
            feed = {input_p: [text_features]}
            score = sess.run(out, feed_dict=feed)

            label = label_id.get(score[0])
            label_d = label_desc.get(label)

            res_list.append(
                json.dumps({"id": id, "label": str(label), "label_desc": label_d}))
    # 写入到文件
    with open(out_path, 'w', encoding='utf-8') as fr:
        for res in res_list:
            fr.write(res)
            fr.write('\n')
    print('predict and write to file over!!!')


if __name__ == '__main__':
    test_file = './data_path/test.json'
    out_path = './data_path/tnews_predict.json'
    predict_main(test_file, out_path)
