import tensorflow as tf
import os, sys
from nnBase import nnBase
import json

default_dir = os.path.abspath(os.curdir)

def create_json_template(filename):
    n_outs = [64, 64, None, 128, 128, None, 256, 256, 256, None, 512, 512, 512, None, 1024, 256]
    # Note that the name of 'layers' must contain '_',
    # and the letter after the last '_' must be integers,
    # indicating the unique index of the layer,
    # the integer will be used as keys to sort the layers.
    layers = [
        'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}',
        'conv_{}', 'conv_{}', 'pool_{}', 'conv_{}', 'conv_{}', 'conv_{}', 'pool_{}',
        'dense_{}', 'dense_{}'
    ]
    layers = [item.format(it) for item, it in zip(layers, range(len(layers)))]
    # activation can be either 'relu' or 'lrelu'
    # loss can be either 'cross_entropy' or 'least_square'
    template = {
        'model_name': 'VGG',
        'layers': {ax: bx for ax, bx in zip(layers, n_outs)},
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_type': 'max',
        'pooling_stride': 2,
        'use_batch_norm': True,
        'is_training': True,
        'activation': 'relu',
        'use_dropout': True,
        'keep_prob': 0.5,
        'num_classes': 10,
        'use_softmax': True,
        'loss': 'cross_entropy',
        'sparse_ys': True,
        'optimizer': 'adam'
    }
    with open(filename, 'w') as file:
        json.dump(template, file, indent=4, sort_keys=True)
    print('New template JSON file in: {}'.format(filename))

def _parse_json_data(json_dir):
    with open(json_dir, 'r') as file:
        args = json.load(file)
    return args


class TFNetworkParser(nnBase):
    '''
    The class takes a JSON file directory as input and construct an neural network with tensorflow.
    Notes:
         The elements in holders do not necessarily be tf.placeholder,
         The interface is for the convenience of taking different types of inputs,
         as long as they are tensorflow types, e.g. tf.DataSet.iterator.get_next_batch.

         The class should only be subclassed when building networks from multiple JSON files are desired,
         e.g. for GAN, the generator and the discriminator should be in two separate JSON files.

         Otherwise, users should either inherit the subclass TFNetworkParserTrain (for training),
         or inherit the subclass TFNetworkParserTest (for testing).
    '''
    name = 'JSON_Parser'
    counter = 0

    def __init__(self, holders, json_dir):
        self.json_data = _parse_json_data(json_dir)

        # If the parameter "use_softmax" is False in JSON, self.y_prob will be None
        self.y_prob, self.y_logits = self.parse_json(holders)
        self.loss = self.objective(holders['ys'])

    def parse_json(self, holders):
        model = holders['xs']
        json_pairs = list(self.json_data['layers'].items())
        json_pairs.sort(key=lambda x: int(x[0].split('_')[-1]))

        with tf.variable_scope(self.json_data['model_name']):
            for name, param in json_pairs:
                model = self.build_from_args(model, name, param)

                # By default, do not apply batch_norm to the fully connected layers
                # So only convolution layer is considered here
                if self.json_data['use_batch_norm'] and 'conv' in name:
                    args = [model, name + '_bn', self.json_data['is_training']]
                    model = self.batch_norm(*args)

                # Activation function
                if self.json_data['activation'].lower() == 'relu':
                    model = tf.nn.relu(model)
                elif self.json_data['activation'].lower() == 'lrelu':
                    model = self.lrelu(model)

                # Similarly, only apply dropout to fully connected layers
                if 'dense' in name and self.json_data['use_dropout']:
                    model = tf.nn.dropout(model, keep_prob=self.json_data['keep_prob'])
            # The last layer is processed separately
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshape_out')
            output = self.fully_connect(model, self.json_data['num_classes'], name='output')
            if self.json_data['use_softmax']:
                prob = tf.nn.softmax(output, name='softmax')
            else:
                prob = None
            return prob, output

    def build_from_args(self, model, name, param):
        if 'conv' in name:
            args = [model, param, name, self.json_data['conv_kernel_size'], self.json_data['conv_kernel_size'],
                    self.json_data['conv_stride'], self.json_data['conv_stride'],
                    not self.json_data['use_batch_norm']]
            return self.conv2d(*args)
        elif 'pool' in name:
            args = [model, name, 2, 2, self.json_data['pooling_stride'],
                    self.json_data['pooling_stride'], self.json_data['pooling_type']]
            return self.pooling(*args)
        elif 'dense' in name:
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshaped')
            return self.fully_connect(model, param, name)
        else:
            raise AttributeError('Invalid input arguments')

    def objective(self, ys):
        if self.json_data['loss'] == 'cross_entropy':
            if self.json_data['sparse_ys']:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys,
                                                                      logits=self.y_logits)
                return tf.reduce_mean(loss)
            else:
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys,
                                                               logits=self.y_logits)
                return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        elif self.json_data['loss'] == 'least_square':
            loss = tf.square(self.y_logits - ys)
            return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        else:
            raise AttributeError('Invalid attribute [loss] in JSON')

    def __del__(self):
        if self.sess is not None:
            self.sess.close()


class TFNetworkParserTest(TFNetworkParser):
    '''
    This class is the TFNetworkParser for testing stage.
    Additional tensorflow builtin accuracy Op is implemented in the class
    for the convenience of testing.

    Notes:
        For testing, the keeo_prob for dropout is set to 1.0
        the mean and variance in batch_norm are set to constants
    '''
    def __init__(self, holders, json_dir, model_dir=default_dir, pre_train=True):
        self.model_dir = model_dir
        super(TFNetworkParserTest, self).__init__(holders, json_dir)
        self.saver = tf.train.Saver()
        arg_ys = tf.argmax(self.y_prob, axis=-1, output_type=tf.int32)
        # the accuracy is only implemented in the subclass for testing
        if self.json_data['sparse_ys']:
            self.accuracy, self.update_acc = tf.metrics.accuracy(holders['ys'], arg_ys)
        else:
            sparse_ys = tf.argmax(holders['ys'], axis=-1, output_type=tf.int32)
            self.accuracy, self.update_acc = tf.metrics.accuracy(sparse_ys, arg_ys)

        # The Session should only be implemented in the subclass
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()
        # the accuracy op contains local variables
        tf.local_variables_initializer().run()

        if pre_train and len(os.listdir(self.model_dir)) != 0:
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    def parse_json(self, holders):
        model = holders['xs']
        json_pairs = list(self.json_data['layers'].items())
        json_pairs.sort(key=lambda x: int(x[0].split('_')[-1]))

        with tf.variable_scope(self.json_data['model_name']):
            for name, param in json_pairs:
                model = self.build_from_args(model, name, param)

                if self.json_data['use_batch_norm'] and 'conv' in name:
                    args = [model, name + '_bn', False]
                    model = self.batch_norm(*args)

                if self.json_data['activation'].lower() == 'relu':
                    model = tf.nn.relu(model)
                elif self.json_data['activation'].lower() == 'lrelu':
                    model = self.lrelu(model)

                # No dropout for testing network (set keep_prob=1.0)
                if 'dense' in name and self.json_data['use_dropout']:
                    model = tf.nn.dropout(model, keep_prob=1.0)
            # The last layer is processed separately
            shape = model.get_shape().as_list()
            if len(shape) != 2:
                model = tf.reshape(model, [shape[0], -1], name='reshape_out')
            output = self.fully_connect(model, self.json_data['num_classes'], name='output')
            if self.json_data['use_softmax']:
                prob = tf.nn.softmax(output, name='softmax')
            else:
                prob = None
            return prob, output


class TFNetworkParserTrain(TFNetworkParser):
    '''
    The class should be subclassed. Users should either override self.train method abstracted in nnBase,
    or implement a training function outside the class.

    Note:
        tensorflow optimizer can be specified in JSON file. The default choice is Adam optimizer.
    '''
    def __init__(self, holders, json_dir, learning_rate=4e-4, model_dir=default_dir, pre_train=True):
        self.model_dir = model_dir
        super(TFNetworkParserTrain, self).__init__(holders, json_dir)
        self.saver = tf.train.Saver()
        arg_ys = tf.argmax(self.y_prob, axis=-1, output_type=tf.int32)
        # To evaluate the temporary accuracy value during training
        if self.json_data['sparse_ys']:
            self.mAP = self._map_op(holders['ys'], arg_ys)
        else:
            sparse_ys = tf.argmax(holders['ys'], axis=-1, output_type=tf.int32)
            self.mAP = self._map_op(sparse_ys, arg_ys)

        # To initialize the optimizer
        train_vars = tf.trainable_variables(self.json_data['model_name'])
        if 'optimizer' in self.json_data:
            if self.json_data['optimizer'].lower() == 'sgd':
                self.optim = tf.train.GradientDescentOptimizer(learning_rate).\
                    minimize(self.loss, var_list=train_vars)
            elif self.json_data['optimizer'].lower() == 'rmsprop':
                self.optim = tf.train.RMSPropOptimizer(learning_rate).\
                    minimize(self.loss, var_list=train_vars)
            else:
                self.optim = tf.train.AdamOptimizer(learning_rate).\
                    minimize(self.loss, var_list=train_vars)
        else:
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=train_vars)

        # To build summary
        loss_sum = tf.summary.scalar('loss', self.loss)
        ap_sum = tf.summary.scalar('temporary average precision', self.mAP)
        self.summary = tf.summary.merge([loss_sum, ap_sum], 'summaries')

        # The Session should only be implemented in the subclass
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        tf.global_variables_initializer().run()

        if pre_train and os.listdir(model_dir):
            _, self.counter = self.load()
        else:
            print('Build model from scratch!!')
            self.counter = 0

    @staticmethod
    def _map_op(label, pred):
        correct = tf.cast(tf.equal(label, pred), dtype=tf.float32)
        return tf.reduce_mean(correct)


if __name__ == '__main__':
    # create_json_template('json2tf.json')
    xs = tf.placeholder(tf.float32, [8, 32, 32, 3], name='xs')
    ys = tf.placeholder(tf.int32, [8], name='ys')
    placeholders = {'xs': xs, 'ys': ys}

    model = TFNetworkParserTrain(placeholders, 'json2tf.json')
    model.show_all_variables()