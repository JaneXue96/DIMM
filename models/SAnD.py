import tensorflow as tf
import tensorflow.contrib as tc
import time
from .nn_module import dense, seq_loss, focal_loss, point_loss
from .attention_module import self_transformer


class SAND(object):
    def __init__(self, args, batch, dim, logger, W, M):
        # logging
        self.logger = logger
        # basic config
        self.cache_W = W
        self.M = M
        self.n_index = dim[0]
        self.n_medicine = dim[1]
        self.n_hidden = 2 * args.n_hidden
        self.use_cudnn = args.use_cudnn
        self.n_batch = tf.get_variable('n_batch', shape=[], dtype=tf.int32, trainable=False)
        self.n_layer = args.n_layer
        self.block_ipt = args.block_ipt
        self.head_ipt = args.head_ipt
        self.n_label = args.n_class
        self.is_map = args.is_map
        self.is_att = args.ipt_att
        self.is_point = args.is_point
        self.is_fc = args.is_fc
        self.opt_type = args.optim
        self.dropout_keep_prob = args.dropout_keep_prob
        self.weight_decay = args.weight_decay

        self.id, self.index, self.medicine, self.seq_len, self.labels = batch.get_next()
        self.N = tf.shape(self.id)[0]
        self.max_len = tf.reduce_max(self.seq_len)
        self.mask = tf.sequence_mask(self.seq_len, self.max_len, dtype=tf.float32, name='masks')
        self.index = tf.slice(self.index, [0, 0, 0], tf.stack([self.N, self.max_len, self.n_index]))
        self.medicine = tf.slice(self.medicine, [0, 0, 0], tf.stack([self.N, self.max_len, self.n_medicine]))
        # self.position = tf.multiply(tf.tile(tf.expand_dims(tf.range(start=1, limit=self.max_len + 1), 0), [self.N, 1]),
        #                             tf.cast(self.mask, dtype=tf.int32))
        self.position = tf.tile(tf.expand_dims(tf.range(start=0, limit=self.max_len), 0), [self.N, 1])
        self.pos_embeddings = tf.Variable(tf.random_normal([720, self.n_hidden], 0.0, self.n_hidden ** -0.5),
                                          trainable=True)
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        # self.lr = tf.train.exponential_decay(args.lr, global_step=self.global_step, decay_steps=args.checkpoint,
        #                                      decay_rate=0.96)
        self.initializer = tc.layers.xavier_initializer()

        self._build_graph()
        # if self.is_train:
        #     # save info
        #     self.saver = tf.train.Saver()
        # else:
        #     self.saver = model_saver

        # initialize the model
        # self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        start_t = time.time()
        self._embedding()
        self._self_attention()
        self._interpolation()
        if self.is_point:
            self._point_label()
        else:
            self._seq_label()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _embedding(self):
        self.input_encodes = tf.concat([self.index, self.medicine], 2)
        with tf.variable_scope('input_embedding', reuse=tf.AUTO_REUSE):
            self.input_emb = tf.layers.conv1d(inputs=self.input_encodes, filters=self.n_hidden, kernel_size=1,
                                              padding='same', kernel_initializer=self.initializer)
        with tf.variable_scope('position_embedding', reuse=tf.AUTO_REUSE):
            self.pos_emb = tf.nn.embedding_lookup(self.pos_embeddings, self.position)
        self.input_encodes = self.input_emb + tf.multiply(self.pos_emb,
                                                          tf.tile(tf.expand_dims(self.mask, axis=2), [1, 1, self.n_hidden]))

        if self.is_train:
            self.input_encodes = tf.nn.dropout(self.input_encodes, self.dropout_keep_prob)

    def _self_attention(self):
        with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
            self.seq_encodes = self_transformer(self.input_encodes, self.input_encodes, self.mask, self.block_ipt,
                                                self.n_hidden, self.head_ipt, self.dropout_keep_prob, True, self.is_train)

    def _interpolation(self):
        with tf.variable_scope('interpolation_embedding', reuse=tf.AUTO_REUSE):
            self.S = tf.transpose(self.seq_encodes, [0, 2, 1])
            self.indices = tf.tile(tf.expand_dims(tf.range(start=0, limit=self.max_len), 0), [self.N, 1])
            self.W = tf.nn.embedding_lookup(self.cache_W, self.indices)
            self.W = tf.multiply(self.W, tf.tile(tf.expand_dims(self.mask, axis=2), [1, 1, self.M]))
            self.U = tf.matmul(self.S, self.W)
            self.inter_emb = tf.reshape(self.U, [self.N, self.n_hidden * self.M])

    def _seq_label(self):
        with tf.variable_scope('seq_labels', reuse=tf.AUTO_REUSE):
            self.seq_encodes = tf.reshape(self.input_encodes, [-1, self.n_hidden])
            # self.label_dense_1 = tf.nn.relu(dense(self.seq_encodes, hidden=int(self.n_hidden / 2), scope='dense_1',
            #                                       initializer=self.initializer))
            # if self.is_train:
            #     self.label_dense_1 = tf.nn.dropout(self.label_dense_1, self.dropout_keep_prob)
            self.outputs = dense(self.seq_encodes, hidden=self.n_label, scope='output_labels',
                                 initializer=self.initializer)
            self.outputs = tf.reshape(self.outputs, tf.stack([-1, self.max_len, self.n_label]))
            self.labels = tf.tile(tf.expand_dims(self.labels, axis=1), tf.stack([1, self.max_len]))

            if self.is_fc:
                self.label_loss = focal_loss(self.outputs, self.labels, self.mask)
            else:
                self.label_loss = seq_loss(self.outputs, self.labels, self.mask)

    def _point_label(self):
        with tf.variable_scope('point_labels', reuse=tf.AUTO_REUSE):
            self.outputs = dense(self.inter_emb, hidden=self.n_label, scope='output_labels',
                                 initializer=self.initializer)
            self.label_loss = point_loss(self.outputs, self.labels)

    def _compute_loss(self):
        self.all_params = tf.trainable_variables()
        self.pre_labels = tf.argmax(self.outputs, axis=1 if self.is_point else 2)
        self.soft_outputs = tf.stop_gradient(tf.nn.softmax(self.outputs))
        self.pre_scores = self.soft_outputs[:, 1]
        self.loss = self.label_loss
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _compute_acc(self):
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.pre_labels, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

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
            # self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.all_params), 25)
            # self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.all_params),
            #                                                global_step=self.global_step)
            # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, 25)
            self.train_op = self.optimizer.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)
