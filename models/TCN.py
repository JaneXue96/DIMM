import tensorflow as tf
import tensorflow.contrib as tc
import time
from .cnn_module import TemporalConvNet
from .nn_module import seq_loss, focal_loss, point_loss


class TCN(object):
    def __init__(self, args, batch, dim, logger):
        # logging
        self.logger = logger
        # basic config
        self.index_size = dim[0]
        self.medicine_size = dim[1]
        self.class_size = args.n_class
        self.channel_size = [args.fsize] * args.levels
        self.is_map = args.is_map
        self.is_point = args.is_point
        self.is_fc = args.is_fc
        self.opt_type = args.optim
        self.dropout_keep_prob = args.dropout_keep_prob
        self.weight_decay = args.weight_decay

        self.id, self.index, self.medicine, self.seq_len, self.org_len, self.labels = batch.get_next()
        self.N = tf.shape(self.id)[0]
        self.max_len = tf.reduce_max(self.seq_len)
        self.mask = tf.sequence_mask(self.seq_len, self.max_len, dtype=tf.float32, name='masks')
        self.index = tf.slice(self.index, [0, 0, 0], tf.stack([self.N, self.max_len, self.index_size]))
        self.medicine = tf.slice(self.medicine, [0, 0, 0], tf.stack([self.N, self.max_len, self.medicine_size]))

        self.n_batch = tf.get_variable('batch_size', shape=[], dtype=tf.int32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # self.lr = tf.train.exponential_decay(args.lr, global_step=self.global_step, decay_steps=args.checkpoint,
        #                                      decay_rate=0.96)
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.initializer = tc.layers.xavier_initializer()

        self._build_graph(args)
        # if self.is_train:
        #     # save info
        #     self.saver = tf.train.Saver()
        # else:
        #     self.saver = model_saver

        # initialize the model
        # self.sess.run(tf.global_variables_initializer())

    def _build_graph(self, args):
        start_t = time.time()
        self._map(args)
        self._tcn(args)
        if self.is_point:
            self._point_label()
        else:
            self._seq_label()
        self._compute_loss()
        # 选择优化算法
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _map(self, args):
        with tf.variable_scope('input_mapping', reuse=tf.AUTO_REUSE):
            if self.is_map:
                with tf.variable_scope('index', reuse=tf.AUTO_REUSE):
                    self.index = tc.layers.fully_connected(self.index, args.n_hidden, activation_fn=None)
                    self.index = tf.reshape(self.index, [-1, args.max_len, args.n_hidden], name='2_3D')
                with tf.variable_scope('medicine', reuse=tf.AUTO_REUSE):
                    self.medicine = tc.layers.fully_connected(self.index, args.n_hidden, activation_fn=None)
                    self.medicine = tf.reshape(self.medicine, [-1, args.max_len, args.n_hidden], name='2_3D')
            self.input_encodes = tf.concat([self.index, self.medicine], 2)
            if self.is_train:
                self.input_encodes = tf.nn.dropout(self.input_encodes, self.dropout_keep_prob)

    def _tcn(self, args):
        with tf.variable_scope('tcn', reuse=tf.AUTO_REUSE):
            self.tcn = TemporalConvNet(input_layer=self.input_encodes, num_channels=self.channel_size,
                                       kernel_size=args.ksize, initializer=self.initializer,
                                       dropout_keep_prob=self.dropout_keep_prob, atten=args.atten, highway=args.highway, gated=args.gated)

    def _seq_label(self):
        with tf.variable_scope('seq_labels', reuse=tf.AUTO_REUSE):
            self.outputs = tc.layers.fully_connected(self.tcn, self.class_size, activation_fn=None)
            self.labels = tf.tile(tf.expand_dims(self.labels, axis=1), tf.stack([1, self.max_len]))

            if self.is_fc:
                self.label_loss = focal_loss(self.outputs, self.labels, self.mask)
            else:
                self.label_loss = seq_loss(self.outputs, self.labels, self.mask)

    def _point_label(self):
        with tf.variable_scope('point_labels', reuse=tf.AUTO_REUSE):
            self.tcn = self.tcn[:, -1, :]
            self.outputs = tc.layers.fully_connected(self.tcn, self.class_size, activation_fn=None)

            self.label_loss = point_loss(self.outputs, self.labels)

    def _compute_loss(self):
        self.all_params = tf.trainable_variables()
        self.pre_labels = tf.argmax(self.outputs, axis=1 if self.is_point else 2)
        self.loss = self.label_loss
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.opt_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.opt_type == 'adam':
                self.optimizer = tc.opt.LazyAdamOptimizer(self.lr)
            elif self.opt_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.opt_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.opt_type))
            self.train_op = self.optimizer.minimize(self.loss)

