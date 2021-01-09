"""BERT finetuning runner for ner （sequence label classification）."""

import collections
import os
import json
import tensorflow as tf

import bert_master.modeling as modeling
import bert_master.optimization as optimization
import bert_master.tokenization as tokenization

flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", './data_path/clue_ner/',
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", './pre_trained/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'cluener',
                    "The name of the task to train.")

flags.DEFINE_string("vocab_file", './pre_trained/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", './model_ckpt/clue_ner/',
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

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 1.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, tag=None):
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
        self.tag = tag


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
                 tag_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_ids = tag_ids
        self.is_real_example = is_real_example


class NerProcessor:
    def get_train_examples(self, data_dir):
        """获取训练集."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """获取验证集."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """获取测试集."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.json")), "test")

    def get_tags(self):
        """填写tag的标签，采用BIO形式标注"""
        # 会在convert_single_example方法中添加头，成为BIO形式标签
        return ['address', 'book', 'company', 'game', 'government',
                'movie', 'name', 'organization', 'position', 'scene']

    def _read_tsv(self, input_file):
        """读取数据集"""
        with open(input_file, encoding='utf-8') as fr:
            lines = fr.readlines()
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if set_type == 'test':
                json_str = json.loads(line)
                text_a = tokenization.convert_to_unicode(json_str['text'])
                tag = None
                guid = json_str['id']
            else:
                text_tag = line.split('\t')
                guid = "%s-%s" % (set_type, i)
                text_a = tokenization.convert_to_unicode(text_tag[0])
                tag = tokenization.convert_to_unicode(text_tag[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, tag=tag))
        return examples


def convert_single_example(ex_index, example, tag_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            tag_ids=[0] * max_seq_length,
            is_real_example=False)

    tag_map = {'O': 0}
    for tag in tag_list:
        tag_b = 'B-' + tag
        tag_i = 'I-' + tag
        tag_map[tag_b] = len(tag_map)
        tag_map[tag_i] = len(tag_map)

    # 因为CLUE要求提交文件中包含索引，所以不能直接使用tokenizer去分割text
    tokens_a = []
    text_list = list(example.text_a)
    for word in text_list:
        token = tokenizer.tokenize(word)
        tokens_a.extend(token)

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

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    if example.tag:
        tag_ids = [0]  # input第一位是[CLS]
        tags = example.tag.strip().split(' ')
        for tag in tags:
            tag_ids.append(tag_map.get(tag))
        tag_ids.append(0)  # input最后一位是[SEP]
    else:
        tag_ids = [0] * max_seq_length
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # test的时候已经*max_len所以不需要再继续padding
        if example.tag:
            tag_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(tag_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in tag_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        tag_ids=tag_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, tag_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, tag_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["tag_ids"] = create_int_feature(feature.tag_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tag_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 tags, num_tags, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # 用bert的sequence输出层
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value
    seq_len = output_layer.shape[1].value
    # [batch, seq_len, emb_size]  16 128 768

    output_weights = tf.get_variable(
        "output_weights", [num_tags, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_tags], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        # 进行matmul需要reshape
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        # [batch*seq_len, num_tags]
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [-1, seq_len, num_tags])

        # 真实的长度
        input_m = tf.count_nonzero(input_mask, -1)
        log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(
            logits, tags, input_m)
        loss = tf.reduce_mean(-log_likelihood)

        # 使用crf_decode输出
        viterbi_sequence, _ = tf.contrib.crf.crf_decode(
            logits, transition_matrix, input_m)

        return loss, logits, viterbi_sequence


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        tag_ids = features["tag_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(tag_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, logits, predictions = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, tag_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # 添加loss的hook，不然在GPU/CPU上不打印loss
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, tag_ids, is_real_example):
                # 这里使用的accuracy来计算，宽松匹配方法
                accuracy = tf.metrics.accuracy(
                    labels=tag_ids, predictions=predictions, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                }

            eval_metrics = (metric_fn,
                            [total_loss, tag_ids, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"predictions": predictions},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cluener": NerProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    #     raise ValueError(
    #         "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tag_list = processor.get_tags()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    # num_labels=2 * len(tag_list) + 1 BI两种外加一个O
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2*len(tag_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, tag_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.data_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, tag_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # label dict的设置
        tag_ids = {0: 'O', 1: 'B-address', 2: 'I-address', 3: 'B-book', 4: 'I-book',
                   5: 'B-company', 6: 'I-company', 7: 'B-game', 8: 'I-game',
                   9: 'B-government', 10: 'I-government', 11: 'B-movie', 12: 'I-movie',
                   13: 'B-name', 14: 'I-name', 15: 'B-organization', 16: 'I-organization',
                   17: 'B-position', 18: 'I-position', 19: 'B-scene', 20: 'I-scene'}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        test_file = os.path.join(FLAGS.data_dir, "test.tf_record")
        file_based_convert_examples_to_features(predict_examples, tag_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                test_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        results = estimator.predict(input_fn=predict_input_fn)

        output_file = os.path.join(FLAGS.data_dir, 'clue_predict.json')
        with open(output_file, 'w', encoding='utf-8') as fr:
            for example, result in zip(predict_examples, results):
                pre_id = result['predictions']
                # print(f'text is {example.text_a}')
                # print(f'preid is {pre_id}')
                text = example.text_a
                # 只获取text中的长度的tag输出
                tags = [tag_ids[tag] for tag in pre_id][1:len(text) + 1]
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
                output_line = json.dumps({'id': example.guid, 'label': pres}, ensure_ascii=False) + '\n'
                fr.write(output_line)


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


if __name__ == "__main__":
    main()
