# attention with maxpooling
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class Attention(Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        print('input_shape', input_shape)
        # input()

        with tf.variable_scope('ww',reuse=tf.AUTO_REUSE):

            self.W = tf.get_variable("ww", shape=(input_shape[-1], 1), initializer=tf.contrib.layers.xavier_initializer())

        self.b = tf.Variable(tf.constant(0.1, shape=[1]), name="bw")

        # self.W = self.add_weight(name="att_weight", shape=[input_shape[-1], 1],
        #                         initializer="normal")
        # self.b = self.add_weight(name="att_bias", shape=[input_shape[1], 1],
        #                         initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


class TextCNN(object):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, \
                 word_seq_len, char_seq_len, embedding_size, l2_reg_lambda=0, \
                 filter_sizes=[3, 4, 5, 6], mode=0):
        if mode == 4 or mode == 5:
            self.input_x_char = tf.placeholder(tf.int32, [None, None, None], name="input_x_char")
            self.input_x_char_pad_idx = tf.placeholder(tf.float32, [None, None, None, embedding_size],
                                                       name="input_x_char_pad_idx")
        if mode == 4 or mode == 5 or mode == 2 or mode == 3:
            self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        if mode == 1 or mode == 3 or mode == 5:
            self.input_x_char_seq = tf.placeholder(tf.int32, [None, 200], name="input_x_char_seq")

        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        att_mechanism = Attention(return_sequences=True)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            if mode == 4 or mode == 5:
                self.char_w = tf.Variable(tf.random_uniform([char_ngram_vocab_size, embedding_size], -1.0, 1.0),
                                          name="char_emb_w")
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.word_w = tf.Variable(tf.random_uniform([word_ngram_vocab_size, embedding_size], -1.0, 1.0),
                                          name="word_emb_w")
            if mode == 1 or mode == 3 or mode == 5:
                self.char_seq_w = tf.Variable(tf.random_uniform([char_vocab_size, embedding_size], -1.0, 1.0),
                                              name="char_seq_emb_w")

            if mode == 4 or mode == 5:
                self.embedded_x_char = tf.nn.embedding_lookup(self.char_w, self.input_x_char)
                self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)
            if mode == 2 or mode == 3 or mode == 4 or mode == 5:
                self.embedded_x_word = tf.nn.embedding_lookup(self.word_w, self.input_x_word)
            if mode == 1 or mode == 3 or mode == 5:
                self.embedded_x_char_seq = tf.nn.embedding_lookup(self.char_seq_w, self.input_x_char_seq)

            if mode == 4 or mode == 5:
                self.sum_ngram_x_char = tf.reduce_sum(self.embedded_x_char, 2)
                self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)

            if mode == 4 or mode == 5:
                self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
            if mode == 2 or mode == 3:
                self.sum_ngram_x_expanded = tf.expand_dims(self.embedded_x_word, -1)
            if mode == 1 or mode == 3 or mode == 5:
                self.char_x_expanded = tf.expand_dims(self.embedded_x_char_seq, -1)

                ########################### WORD CONVOLUTION LAYER ################################
        if mode == 2 or mode == 3 or mode == 4 or mode == 5:
            pooled_x = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.sum_ngram_x_expanded,
                        w,
                        strides=[1, 1, 32, 1],
                        padding="SAME",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h_attention_word = att_mechanism(tf.reshape(h, [-1, 200, h.shape[-2] * h.shape[-1]]))
                    h_attention_word = tf.reshape(h_attention_word, [-1, 200, 1, h_attention_word.shape[-1]])
                    pooled = tf.nn.max_pool(
                        h_attention_word,
                        ksize=[1, word_seq_len, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_x.append(pooled)

            num_filters_total = 256 * len(filter_sizes)
            self.h_pool = tf.concat(pooled_x, 3)
            self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")
            self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x")

            ########################### CHAR CONVOLUTION LAYER ###########################
        if mode == 1 or mode == 3 or mode == 5:
            pooled_char_x = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, 256]
                    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                    conv = tf.nn.conv2d(
                        self.char_x_expanded,
                        w,
                        strides=[1, 1, 32, 1],
                        padding="SAME",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # print("h_shape",h.shape)
                    h_attention = att_mechanism(tf.reshape(h, [-1, 200, h.shape[-2]*h.shape[-1]]))
                    h_attention = tf.reshape(h_attention, [-1, 200, 1, h_attention.shape[-1]])

                    # print('h_attention',h_attention.shape)

                    pooled = tf.nn.max_pool(
                        h_attention,
                        ksize=[1, char_seq_len, 1, 1],
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="pool")
                    pooled_char_x.append(pooled)
                    # print('pooled_char_x',pooled_char_x[0].shape)
            num_filters_total = 256 * len(filter_sizes)
            self.h_char_pool = tf.concat(pooled_char_x, -1)
            # self.h_char_pool = tf.concat(pooled_char_x, 2)
            # print("h_char_pool", self.h_char_pool.shape)
            # num_filters_total = self.h_char_pool.shape[-2] * self.h_char_pool.shape[-1] #256*len(filter_sizes)
            self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total], name="pooled_char_x")
            self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob, name="dropout_char_x")
            # print("char_h_drop",self.char_h_drop.shape)
            # input()

        ############################### CONCAT WORD AND CHAR BRANCH ############################
        if mode == 3 or mode == 5:
            with tf.name_scope("word_char_concat"):

                ww = tf.get_variable("ww", shape=(num_filters_total, 512),
                                     initializer=tf.contrib.layers.xavier_initializer())
                bw = tf.Variable(tf.constant(0.1, shape=[512]), name="bw")
                l2_loss += tf.nn.l2_loss(ww)
                l2_loss += tf.nn.l2_loss(bw)
                word_output = tf.nn.xw_plus_b(self.h_drop, ww, bw)

                wc = tf.get_variable("wc", shape=(num_filters_total, 512),
                                     initializer=tf.contrib.layers.xavier_initializer())
                bc = tf.Variable(tf.constant(0.1, shape=[512]), name="bc")
                l2_loss += tf.nn.l2_loss(wc)
                l2_loss += tf.nn.l2_loss(bc)
                char_output = tf.nn.xw_plus_b(self.char_h_drop, wc, bc)

                self.conv_output = tf.concat([word_output, char_output], 1)
        elif mode == 2 or mode == 4:
            self.conv_output = self.h_drop
        elif mode == 1:
            self.conv_output = self.char_h_drop

        ################################ RELU AND FC ###################################
        with tf.name_scope("output"):
            w0 = tf.get_variable("w0", shape=[num_filters_total, 512],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b0")
            l2_loss += tf.nn.l2_loss(w0)
            l2_loss += tf.nn.l2_loss(b0)
            output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)

            w1 = tf.get_variable("w1", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1")
            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)
            output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)

            w2 = tf.get_variable("w2", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
            l2_loss += tf.nn.l2_loss(w2)
            l2_loss += tf.nn.l2_loss(b2)
            output2 = tf.nn.relu(tf.matmul(output1, w2) + b2)

            w = tf.get_variable("w", shape=(128, 2), initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(output2, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy") 
