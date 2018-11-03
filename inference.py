import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import warnings
import time
import os
import logging
from models.rnn_module import cu_rnn
from models.nn_module import dense, seq_loss
from models.attention_module import self_transformer

warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

logger = logging.getLogger('Medical')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

data_dir = 'data/raw_data/inference'
model_dir = 'multi_task/DIMM_final/models'
n_index = 205
n_medicine = 241


class InfModel(object):
    def __init__(self):
        self.id = tf.placeholder(tf.int64, [None])
        self.index = tf.placeholder(tf.float32, [2, n_index])
        self.medicine = tf.placeholder(tf.float32, [2, n_medicine])
        self.seq_len = tf.placeholder(tf.int64, [None])
        self.labels = tf.placeholder(tf.int64, [None])

        self.n_hidden = 64
        self.n_batch = 2
        self.n_layer = 2
        self.n_label = 2
        self.N = tf.shape(self.id)[0]
        self.max_len = tf.reduce_max(self.seq_len)
        self.mask = tf.sequence_mask(self.seq_len, self.max_len, dtype=tf.float32, name='masks')
        self.padding = tf.sequence_mask(self.seq_len, self.max_len, dtype=tf.int32, name='padding')
        self.index = tf.slice(self.index, [0, 0, 0], tf.stack([self.N, self.max_len, n_index]))
        self.medicine = tf.slice(self.medicine, [0, 0, 0], tf.stack([self.N, self.max_len, n_medicine]))
        self.initializer = tc.layers.xavier_initializer()
        self._build_graph()

    def _build_graph(self):
        start_t = time.time()
        self._encode()
        self._rnn()
        self._step_attention()
        self._seq_label()
        self._compute_loss()
        logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _encode(self):
        with tf.variable_scope('input_encoding', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('index', reuse=tf.AUTO_REUSE):
                self.index = dense(self.index, hidden=self.n_hidden, initializer=self.initializer)
                self.index = tf.reshape(self.index, [-1, self.max_len, self.n_hidden], name='2_3D')
            with tf.variable_scope('medicine', reuse=tf.AUTO_REUSE):
                self.medicine = dense(self.medicine, hidden=self.n_hidden, initializer=self.initializer)
                self.medicine = tf.reshape(self.medicine, [-1, self.max_len, self.n_hidden], name='2_3D')
            self.i2m = self._input_attention(self.index, self.medicine, self.n_hidden, 'i2m_attention')
            self.m2i = self._input_attention(self.medicine, self.index, self.n_hidden, 'm2i_attention')
            self.index = self._input_attention(self.index, self.index, self.n_hidden, 'i2i_attention')
            self.medicine = self._input_attention(self.medicine, self.medicine, self.n_hidden, 'm2m_attention')
            self.input_encodes = tf.concat([self.index, self.medicine, self.i2m, self.m2i], 2)
            self.input_encodes = tf.nn.dropout(self.input_encodes, 1.0)

    def _input_attention(self, input_x, input_y, n_unit, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_encodes = self_transformer(input_x, input_y, self.mask, 4, n_unit, 1, 1.0, False, True)
            return input_encodes

    def _rnn(self):
        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            self.seq_encodes, _ = cu_rnn('bi-gru', self.input_encodes, self.n_hidden, self.n_batch, False, self.n_layer)
        self.n_hidden *= self.n_layer
        self.seq_encodes = tf.nn.dropout(self.seq_encodes, 1.0)

    def _step_attention(self):
        with tf.variable_scope('step_attention', reuse=tf.AUTO_REUSE):
            self.seq_encodes = self_transformer(self.seq_encodes, self.seq_encodes, self.mask, 4, self.n_hidden,
                                                4, 1.0, True, True)

    def _seq_label(self):
        with tf.variable_scope('seq_labels', reuse=tf.AUTO_REUSE):
            self.seq_encodes = tf.reshape(self.seq_encodes, [-1, self.n_hidden])
            self.outputs = dense(self.seq_encodes, hidden=self.n_label, scope='output_labels',
                                 initializer=self.initializer)
            self.outputs = tf.reshape(self.outputs, tf.stack([-1, self.max_len, self.n_label]))
            self.labels = tf.tile(tf.expand_dims(self.labels, axis=1), tf.stack([1, self.max_len]))
            self.label_loss = seq_loss(self.outputs, self.labels, self.mask)

    def _compute_loss(self):
        self.all_params = tf.trainable_variables()
        self.soft_outputs = tf.stop_gradient(tf.nn.softmax(self.outputs))
        self.pre_labels = tf.argmax(self.outputs, 2)
        self.pre_scores = self.outputs[:, :, 1]
        self.loss = self.label_loss


class Inference(object):
    def __init__(self):
        self.model = InfModel()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def response(self, context, question):
        sess = self.sess
        model = self.model
        span, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs = \
            self.prepro(context, question)
        yp1, yp2 = \
            sess.run(
                [model.yp1, model.yp2],
                feed_dict={
                    model.c: context_idxs, model.q: ques_idxs,
                    model.ch: context_char_idxs, model.qh: ques_char_idxs,
                    model.tokens_in_context: len(span)})
        start_idx = span[yp1[0]][0]
        end_idx = span[yp2[0]][1]
        return context[start_idx: end_idx]

    def prepro(self, context, question):
        context = context.replace("''", '" ').replace("``", '" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        ques = question.replace("''", '" ').replace("``", '" ')
        ques_tokens = word_tokenize(ques)
        ques_chars = [list(token) for token in ques_tokens]

        context_idxs = np.zeros([1, len(context_tokens)], dtype=np.int32)
        context_char_idxs = np.zeros(
            [1, len(context_tokens), char_limit], dtype=np.int32)
        ques_idxs = np.zeros([1, len(ques_tokens)], dtype=np.int32)
        ques_char_idxs = np.zeros(
            [1, len(ques_tokens), char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word2idx_dict:
                    return self.word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in self.char2idx_dict:
                return self.char2idx_dict[char]
            return 1

        for i, token in enumerate(context_tokens):
            context_idxs[0, i] = _get_word(token)

        for i, token in enumerate(ques_tokens):
            ques_idxs[0, i] = _get_word(token)

        for i, token in enumerate(context_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[0, i, j] = _get_char(char)

        for i, token in enumerate(ques_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[0, i, j] = _get_char(char)
        return spans, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs


if __name__ == "__main__":
    infer = Inference()
    context = "In meteorology, precipitation is any product of the condensation " \
              "of atmospheric water vapor that falls under gravity. The main forms " \
              "of precipitation include drizzle, rain, sleet, snow, graupel and hail." \
              "Precipitation forms as smaller droplets coalesce via collision with other " \
              "rain drops or ice crystals within a cloud. Short, intense periods of rain " \
              "in scattered locations are called “showers”."
    ques1 = "What causes precipitation to fall?"
    ques2 = "What is another main form of precipitation besides drizzle, rain, snow, sleet and hail?"
    ques3 = "Where do water droplets collide with ice crystals to form precipitation?"

    # Correct: gravity, Output: drizzle, rain, sleet, snow, graupel and hail
    ans1 = infer.response(context, ques1)
    print("Answer 1: {}".format(ans1))

    # Correct: graupel, Output: graupel
    ans2 = infer.response(context, ques2)
    print("Answer 2: {}".format(ans2))

    # Correct: within a cloud, Output: within a cloud
    ans3 = infer.response(context, ques3)
    print("Answer 3: {}".format(ans3))