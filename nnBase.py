import tensorflow as tf
from tensorflow.contrib import slim
import os, re


def _check_valid(func):
    def decorate(this):
        info = 'Parameters like saver, sess, counter or model_dir should be specified'
        if not this._isvalid():
            raise NotImplementedError(info)
        return func(this)
    return decorate


class nnBase(object):
    saver = None
    sess = None
    name = 'base'
    counter = None
    model_dir = None

    def _isvalid(self):
        return self.saver is not None \
               and self.sess is not None \
               and self.model_dir is not None

    @staticmethod
    def batch_norm(input_op, name, is_training, epsilon=1e-5, momentum=0.99):
        return tf.contrib.layers.batch_norm(input_op, decay=momentum, updates_collections=None,
                                            epsilon=epsilon, scale=True, is_training=is_training, scope=name)

    @staticmethod
    def lrelu(input_op, leak=0.2, name='lrelu'):
        return tf.maximum(input_op, leak * input_op, name=name)

    @staticmethod
    def conv2d(input_op, n_out, name, kh=5, kw=5, dh=2, dw=2, bias=True):
        '''
        :param bias: This parameter should be specified as False when using batch norm!!
         The bias is a fake parameter for batch norm due to the re-normalization
        '''
        n_in = input_op.get_shape()[-1].value
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernel_2d_w',
                                     shape=(kh, kw, n_in, n_out),
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
            if bias:
                biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
                return tf.nn.bias_add(conv, biases)
            else:
                return conv

    @staticmethod
    def conv1d(input_op, n_out, name, ksize=5, dsize=1, bias=True):
        '''
        :param input_op: the input size should be [batch_size, sequence_length, channel_size].
        :param bias: the parameter should be specified as False when
         using batch_norm before ReLU, setting it to True will not make
         the program fail but waste the memory.
        '''
        n_in = input_op.get_shape()[-1].value
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernel_1d_w',
                                     shape=(ksize, n_in, n_out),
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv1d(input_op, kernel, dsize, padding='SAME')
            if bias:
                biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
                return tf.nn.bias_add(conv, biases)
            else:
                return conv

    @staticmethod
    def deconv2d(input_op, output_shape, kh=5, kw=5, dh=2, dw=2, name='deconv', bias_init=0.0):
        n_in = input_op.get_shape()[-1].value
        n_out = output_shape[-1]
        # filter : [height, width, output_channels, in_channels]
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernels',
                                     shape=(kh, kw, n_out, n_in),
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                            output_shape=output_shape,
                                            strides=(1, dh, dw, 1))
            biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
            return tf.nn.bias_add(deconv, biases)

    @staticmethod
    def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
        if 'max' in pooling_type:
            return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
        else:
            return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

    @staticmethod
    def pooling1d(input_op, name, size=2):
        return tf.layers.max_pooling1d(input_op, pool_size=size, strides=size, name=name)

    @staticmethod
    def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0):
        '''
        :return: The tf.matmul op is suggested to be time consuming.
         This is still the traditional implementation using tf.matmul
         The alternative implementation can be found in self.dense()
        '''
        n_in = input_op.get_shape()[-1].value
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='weights',
                                     shape=[n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
            return tf.matmul(input_op, kernel) + biases

    @staticmethod
    def dense(input_op, n_out, name='dense', bias_init=0.0):
        '''
         Since tf.matmul is a time-consuming op,
         A better solution is using element-wise multiply, reduce_sum and reshape
         ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
         element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
         reshape to [a, c]
        '''
        raise NotImplementedError('Remained')

    @staticmethod
    def atrous_conv(input_op, n_out, rate, name, kh=3, kw=3, activate='relu'):
        n_in = input_op.get_shape()[-1].value
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='kernel_w',
                                     shape=(kh, kw, n_in, n_out),
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.atrous_conv2d(input_op, kernel, rate=rate, padding='SAME')
            biases = tf.get_variable('biases', (n_out), initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def objective(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def show_all_variables(*args, **kwargs):
        all_variables = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(all_variables, print_info=True)

    @_check_valid
    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif len(os.listdir(self.model_dir)) != 0:
            fs = os.listdir(self.model_dir)
            for f in fs:
                os.remove(self.model_dir + f)
        save_path = self.saver.save(self.sess, self.model_dir + self.name + '.model',
                                    global_step=self.counter)
        print('MODEL RESTORED IN: ' + save_path)

    @_check_valid
    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, self.model_dir + ckpt_name)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == '__main__':
    net = nnBase()
    net.show_all_variables()
    net.load()