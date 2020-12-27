""" BERT finetuning runner for multi-learning task """

import collections
import math
import os
import random
import pandas as pd

import bert_master.modeling as modeling
import bert_master.optimization as optimization
import bert_master.tokenization as tokenization
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", './data_path/',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", './pre_trained/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", './pre_trained/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", './model_ckpt/multi_learning/',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", './pre_trained/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 1,
                     "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, task=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.task = task


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 task,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.task = task
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, task):
        data = pd.read_csv(input_file, sep='\t', encoding='utf-8')
        if task == 'nli':
            data.columns = ['id', 'texta', 'textb', 'label']
        else:
            data.columns = ['id', 'text', 'label']
        lines = []
        for index, row in data.iterrows():
            if task == 'nli':
                lines.append((row['texta'], row['textb'], row['label']))
            else:
                lines.append((row['text'], row['label']))
        return lines


class AllProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        emotion_dir = os.path.join(data_dir, 'train_emotion.csv')
        news_dir = os.path.join(data_dir, 'train_news.csv')
        nli_dir = os.path.join(data_dir, 'train_nli.csv')
        emotion_lines = self._read_csv(emotion_dir, 'emotion')
        news_lines = self._read_csv(news_dir, 'news')
        nli_lines = self._read_csv(nli_dir, 'nli')
        return self._create_examples(emotion_lines, news_lines, nli_lines, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        emotion_dir = os.path.join(data_dir, 'dev_emotion.csv')
        news_dir = os.path.join(data_dir, 'dev_news.csv')
        nli_dir = os.path.join(data_dir, 'dev_nli.csv')
        emotion_lines = self._read_csv(emotion_dir, 'emotion')
        news_lines = self._read_csv(news_dir, 'news')
        nli_lines = self._read_csv(nli_dir, 'nli')
        return self._create_examples(emotion_lines, news_lines, nli_lines, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        pass

    def get_labels(self):
        """See base class."""
        return [['sadness', 'anger', 'happiness', 'fear', 'like',
                 'disgust', 'surprise'],
                ['108', '104', '106', '112', '109', '103', '116', '101',
                 '107', '100', '102', '110', '115', '113', '114'],
                ['0', '1', '2']]

    def _create_examples(self, emotion_lines, news_lines, nli_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        # emotion
        for (i, line) in enumerate(emotion_lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[0])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='1'))

        # news
        for i, line in enumerate(news_lines):
            guid = f'news_{set_type}_{i}'
            if set_type == 'test':
                text_a = tokenization.convert_to_unicode(line[0])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(str(line[1]))

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, task='2'))

        # nli
        for i, line in enumerate(nli_lines):
            guid = f'news_{set_type}_{i}'
            if set_type == 'test':
                text_a = tokenization.convert_to_unicode(line[0])
                text_b = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                text_b = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(str(line[2]))

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, task='3'))

        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            task=0,
            is_real_example=False)

    emotion_label_map = {}
    news_label_map = {}
    nli_label_map = {}
    for (i, label) in enumerate(label_list[0]):
        emotion_label_map[label] = i
    for (i, label) in enumerate(label_list[1]):
        news_label_map[label] = i
    for (i, label) in enumerate(label_list[2]):
        nli_label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

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

    task = example.task
    if task == '1': label_id = emotion_label_map[example.label]
    if task == '2': label_id = news_label_map[example.label]
    if task == '3': label_id = nli_label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        task=int(task),
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, type):
    """Convert a set of `InputExample`s to a TFRecord file."""

    emotion_out = os.path.join(output_file, f'emotion_{type}.record')
    news_out = os.path.join(output_file, f'news_{type}.record')
    nli_out = os.path.join(output_file, f'nli_{type}.record')

    emotion_writer = tf.python_io.TFRecordWriter(emotion_out)
    news_writer = tf.python_io.TFRecordWriter(news_out)
    nli_writer = tf.python_io.TFRecordWriter(nli_out)

    emotion_cnt = 0
    news_cnt = 0
    nli_cnt = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["task"] = create_int_feature([feature.task])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        if feature.task == 1:
            emotion_cnt += 1
            emotion_writer.write(tf_example.SerializeToString())
        if feature.task == 2:
            news_cnt += 1
            news_writer.write(tf_example.SerializeToString())
        if feature.task == 3:
            nli_cnt += 1
            nli_writer.write(tf_example.SerializeToString())

    emotion_writer.close()
    news_writer.close()
    nli_writer.close()
    print(f'the emotion news nli cnt is {emotion_cnt} {news_cnt} {nli_cnt}')
    return emotion_cnt, news_cnt, nli_cnt


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


# todo 三个全连接层，然后分别对应不同的任务
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings, task):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    # 三个任务对应的三个全连接层参数
    emotion_weights = tf.get_variable(
        "emotion_weights", [7, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    emotion_bias = tf.get_variable(
        "emotion_bias", [7], initializer=tf.zeros_initializer())

    news_weights = tf.get_variable(
        "news_weights", [15, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    news_bias = tf.get_variable(
        "news_bias", [15], initializer=tf.zeros_initializer())

    nli_weights = tf.get_variable(
        "nli_weights", [3, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    nli_bias = tf.get_variable(
        "nli_bias", [3], initializer=tf.zeros_initializer())

    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    emotion_logits = tf.matmul(output_layer, emotion_weights, transpose_b=True)
    emotion_logits = tf.nn.bias_add(emotion_logits, emotion_bias)

    news_logits = tf.matmul(output_layer, news_weights, transpose_b=True)
    news_logits = tf.nn.bias_add(news_logits, news_bias)

    nli_logits = tf.matmul(output_layer, nli_weights, transpose_b=True)
    nli_logits = tf.nn.bias_add(nli_logits, nli_bias)

    logits = tf.cond(
        tf.equal(task, 1),
        lambda: emotion_logits,
        lambda: tf.cond(tf.equal(task, 2), lambda: news_logits, lambda: nli_logits)
    )
    depth = tf.cond(
        tf.equal(task, 1),
        lambda: 7,
        lambda: tf.cond(tf.equal(task, 2), lambda: 15, lambda: 3)
    )

    predictions = tf.argmax(logits, axis=-1, output_type=tf.int64, name='pre_id')

    with tf.variable_scope("loss"):
        # probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=depth, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        # todo 计算acc
        equals = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.int64))
        acc = equals / FLAGS.eval_batch_size
        return loss, logits, acc


def get_input_data(input_file, seq_len, batch_size, is_training):
    def parser(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_len], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_len], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_len], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
        }
        # 解析的时候需要的是int64
        example = tf.parse_single_example(record, features=name_to_features)
        input_ids = example["input_ids"]
        input_mask = example["input_mask"]
        segment_ids = example["segment_ids"]
        labels = example["label_ids"]
        return input_ids, input_mask, segment_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    if is_training:
        dataset = dataset.map(parser).batch(batch_size).shuffle(buffer_size=3000)
    else:
        dataset = dataset.map(parser).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, labels = iterator.get_next()
    return input_ids, input_mask, segment_ids, labels


def main():
    """ 训练主入口 """
    tf.logging.info('start to train')

    # 部分参数设置
    process = AllProcessor()
    label_list = process.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = process.get_train_examples(FLAGS.data_dir)
    train_cnt = file_based_convert_examples_to_features(
        train_examples,
        label_list,
        FLAGS.max_seq_length,
        tokenizer,
        FLAGS.data_dir,
        'train'
    )
    dev_examples = process.get_dev_examples(FLAGS.data_dir)
    dev_cnt = file_based_convert_examples_to_features(
        dev_examples,
        label_list,
        FLAGS.max_seq_length,
        tokenizer,
        FLAGS.data_dir,
        'dev'
    )

    # 输入输出定义
    input_ids = tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length],
                               name='input_ids')
    input_mask = tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length],
                                name='input_mask')
    segment_ids = tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length],
                                 name='segment_ids')
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    task = tf.placeholder(tf.int64, name='task')

    # bert相关参数设置
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # todo 模型create_model
    loss, logits, acc = create_model(
        bert_config,
        True,
        input_ids,
        input_mask,
        segment_ids,
        labels,
        False,
        task
    )
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size)
    num_warmup_steps = math.ceil(
        num_train_steps * FLAGS.train_batch_size * FLAGS.warmup_proportion)
    train_op = optimization.create_optimizer(
        loss,
        FLAGS.learning_rate,
        num_train_steps * FLAGS.num_train_epochs,
        num_warmup_steps,
        False
    )

    # 初始化参数
    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(
        [v for v in tf.global_variables()
         if 'adam_v' not in v.name and 'adam_m' not in v.name])

    with tf.Session() as sess:
        sess.run(init_global)
        print('start to load bert params')
        if FLAGS.init_checkpoint:
            # tvars = tf.global_variables()
            tvars = tf.trainable_variables()
            print("global_variables", len(tvars))
            assignment_map, initialized_variable_names = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            FLAGS.init_checkpoint)
            print("initialized_variable_names:", len(initialized_variable_names))
            saver_ = tf.train.Saver([v for v in tvars if v.name in initialized_variable_names])
            saver_.restore(sess, FLAGS.init_checkpoint)
            tvars = tf.global_variables()
            initialized_vars = [v for v in tvars if v.name in initialized_variable_names]
            not_initialized_vars = [v for v in tvars if v.name not in initialized_variable_names]
            print('all size %s; not initialized size %s' % (len(tvars), len(not_initialized_vars)))
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            # for v in initialized_vars:
            #     print('initialized: %s, shape = %s' % (v.name, v.shape))
            # for v in not_initialized_vars:
            #     print('not initialized: %s, shape = %s' % (v.name, v.shape))
        else:
            print('the bert init checkpoint is None!!!')
            sess.run(tf.global_variables_initializer())

        # 训练的step
        def train_step(ids, mask, seg, true_y, task_id):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: seg,
                    labels: true_y,
                    task: task_id}
            _, logits_out, loss_out = sess.run([train_op, logits, loss], feed_dict=feed)
            return logits_out, loss_out

        # 验证的step
        def dev_step(ids, mask, seg, true_y, task_id):
            feed = {input_ids: ids,
                    input_mask: mask,
                    segment_ids: seg,
                    labels: true_y,
                    task: task_id}
            loss_out, acc_out = sess.run([loss, acc], feed_dict=feed)
            return loss_out, acc_out

        # 开始训练
        for epoch in range(FLAGS.num_train_epochs):
            print(f'start to train and the epoch:{epoch}')
            #epoch_loss = do_train(sess, train_cnt, train_step)
            #print(f'the epoch{epoch} loss is {epoch_loss}')
            saver.save(sess, FLAGS.output_dir + 'bert.ckpt', global_step=epoch)
            # 每一个epoch开始验证模型
            # do_eval(sess, dev_cnt, dev_step)

        print('the training is over!!!!')


def set_random_task(train_cnt):
    """ 任务采样 : 各任务每个epoch 迭代的step次数 """
    # emotion cnt
    emotion_cnt = train_cnt[0] // FLAGS.train_batch_size
    news_cnt = train_cnt[1] // FLAGS.train_batch_size
    nli_cnt = train_cnt[2] // FLAGS.train_batch_size

    emotion_list = [1] * emotion_cnt
    news_list = [2] * news_cnt
    nli_list = [3] * nli_cnt

    task_list = emotion_list + news_list + nli_list

    random.shuffle(task_list)

    return task_list


def do_train(sess, train_cnt, train_step):
    """ 模型训练 """
    emotion_train_file = os.path.join(FLAGS.data_dir, 'emotion_train.record')
    news_train_file = os.path.join(FLAGS.data_dir, 'news_train.record')
    nli_train_file = os.path.join(FLAGS.data_dir, 'nli_train.record')
    ids1, mask1, seg1, labels1 = get_input_data(
        emotion_train_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, True)
    ids2, mask2, seg2, labels2 = get_input_data(
        news_train_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, True)
    ids3, mask3, seg3, labels3 = get_input_data(
        nli_train_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, True)

    # 设置任务list
    tasks = set_random_task(train_cnt)

    total_loss = 0
    for step, task_id in enumerate(tasks):
        if task_id == 1:
            ids_train, mask_train, seg_train, y_train = sess.run(
                [ids1, mask1, seg1, labels1])
        if task_id == 2:
            ids_train, mask_train, seg_train, y_train = sess.run(
                [ids2, mask2, seg2, labels2])
        if task_id == 3:
            ids_train, mask_train, seg_train, y_train = sess.run(
                [ids3, mask3, seg3, labels3])

        _, step_loss = train_step(ids_train, mask_train, seg_train, y_train, task_id)

        print(f'the step loss: {step_loss}')

        total_loss += step_loss

    return total_loss / len(tasks)


def do_eval(sess, dev_cnt, dev_step):
    # 数据目录路径
    emotion_dev_file = os.path.join(FLAGS.data_dir, 'emotion_dev.record')
    news_dev_file = os.path.join(FLAGS.data_dir, 'news_dev.record')
    nli_dev_file = os.path.join(FLAGS.data_dir, 'nli_dev.record')

    ids1, mask1, seg1, labels1 = get_input_data(
        emotion_dev_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, False)
    ids2, mask2, seg2, labels2 = get_input_data(
        news_dev_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, False)
    ids3, mask3, seg3, labels3 = get_input_data(
        nli_dev_file, FLAGS.max_seq_length,
        FLAGS.train_batch_size, False)

    # 验证emotion的
    total_dev_acc = 0
    step_cnt = dev_cnt[0] // FLAGS.eval_batch_size
    for step in range(step_cnt):
        ids_dev, mask_dev, seg_dev, y_dev = sess.run(
            [ids1, mask1, seg1, labels1])
        _, dev_acc = dev_step(ids_dev, mask_dev, seg_dev, y_dev, 1)
        total_dev_acc += dev_acc
    print(f'===the emotion acc is {total_dev_acc / step_cnt}===')

    total_dev_acc = 0
    step_cnt = dev_cnt[1] // FLAGS.eval_batch_size
    for step in range(step_cnt):
        ids_dev, mask_dev, seg_dev, y_dev = sess.run(
            [ids2, mask2, seg2, labels2])
        _, dev_acc = dev_step(ids_dev, mask_dev, seg_dev, y_dev, 2)
        total_dev_acc += dev_acc
    print(f'===the news acc is {total_dev_acc / step_cnt}===')

    total_dev_acc = 0
    step_cnt = dev_cnt[3] // FLAGS.eval_batch_size
    for step in range(step_cnt):
        ids_dev, mask_dev, seg_dev, y_dev = sess.run(
            [ids3, mask3, seg3, labels3])
        _, dev_acc = dev_step(ids_dev, mask_dev, seg_dev, y_dev, 3)
        total_dev_acc += dev_acc
    print(f'===the nli acc is {total_dev_acc / step_cnt}===')


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
