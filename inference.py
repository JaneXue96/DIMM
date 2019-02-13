import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import pandas as pd
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
batch_size = 8
max_len = 720


class InfModel(object):
    def __init__(self):
        self.id = tf.placeholder(tf.int32, [None])
        self.index = tf.placeholder(tf.float32, [batch_size, max_len, n_index])
        self.medicine = tf.placeholder(tf.float32, [batch_size, max_len, n_medicine])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None])

        self.n_hidden = 64
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
            input_encodes = self_transformer(input_x, input_y, self.mask, 4, n_unit, 1, 1.0, True, False)
            return input_encodes

    def _rnn(self):
        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            self.seq_encodes, _ = cu_rnn('bi-gru', self.input_encodes, self.n_hidden, batch_size, False, self.n_layer)
        self.n_hidden *= self.n_layer
        self.seq_encodes = tf.nn.dropout(self.seq_encodes, 1.0)

    def _step_attention(self):
        with tf.variable_scope('step_attention', reuse=tf.AUTO_REUSE):
            self.seq_encodes = self_transformer(self.seq_encodes, self.seq_encodes, self.mask, 4, self.n_hidden,
                                                4, 1.0, True, False)

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

    def response(self, file_path):
        sess = self.sess
        model = self.model
        ids, indexes, medicines, seq_lens, labels, names = self.prepro(file_path)
        pre_labels, probs = sess.run([model.pre_labels, model.soft_outputs],
                                     feed_dict={model.id: ids, model.index: indexes, model.medicine: medicines,
                                                model.seq_len: seq_lens, model.labels: labels})

        return pre_labels, probs

    def prepro(self, data_path):
        b_ids = np.zeros(batch_size, dtype=np.int64)
        b_index = np.zeros((batch_size, n_index), dtype=np.float32)
        b_medicine = np.zeros((batch_size, n_medicine), dtype=np.float32)
        b_seq_len = np.zeros(batch_size, dtype=np.int64)
        b_labels = np.zeros(batch_size, dtype=np.int64)
        b_names = []
        for i, file in enumerate(os.listdir(data_path)):
            if file.startswith('0'):
                dead = 0
            else:
                dead = 1
            raw_sample = pd.read_csv(os.path.join(data_path, file), sep=',')
            raw_sample = raw_sample.fillna(0)
            medicine = raw_sample.iloc[:, 209:].as_matrix()
            index = raw_sample.iloc[:, 3:208].as_matrix()
            b_ids[i] = i
            b_index[i] = index
            b_medicine[i] = medicine
            b_seq_len[i] = index.shape[0]
            b_labels[i] = dead
            b_names.append(file)

        return b_ids, b_index, b_medicine, b_seq_len, b_labels, b_names


if __name__ == "__main__":
    infer = Inference()

    labels, probs = infer.response(data_dir)
    print('Labels')
    print(labels)
    print('Probs')
    print(probs)
