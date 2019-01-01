import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1
import numpy as np
import pandas as pd
import os, math
from datetime import datetime
import tabulate

# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
flags.DEFINE_string('output_dir', './results.csv', '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_string('data_dir', '../../datasets/cloud/mode_2004/', '')
flags.DEFINE_string('model_dir', './checkpoints/models/resnet_ordinal.model', '')
flags.DEFINE_string('losstype', 'cross_entropy', '')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_string('model_basedir', './checkpoints/models/', '')
FLAGS = flags.FLAGS
ERROR_FLAG = 0


def save(sess, model_dir, counter):
	saver = tf.train.Saver(max_to_keep=1)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	save_path = saver.save(sess, model_dir, global_step=counter)
	print('MODEL RESTORED IN: ' + save_path)


def load(sess, model_dir):
	import re
	print(' [*] Reading checkpoints...')
	ckpt = tf.train.get_checkpoint_state(model_dir)
	saver = tf.train.Saver(max_to_keep=1)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		saver.restore(sess, model_dir + ckpt_name)
		counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
		print(" [*] Success to read {}".format(ckpt_name))
		return counter
	else:
		print(" [*] Failed to find a checkpoint")
		return ERROR_FLAG

def get_counter(model_dir):
	import re
	print(' [*] Try reading global counter...')
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and ckpt.model_checkpoint_path:
		ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		print(" [*] Success to read {}".format(ckpt_name))
		return int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
	else:
		print(" [*] Failed to find a global counter")
		return ERROR_FLAG


def init_reader(path=FLAGS.data_dir, batch_size=8, epoch=10, is_training=True):
	def _parse_function(xs, ys):
		x_img_str = tf.read_file(xs)
		x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str), tf.float32)
		x_img_resized = tf.image.resize_images(x_img_decoded, size=[512, 512],
		                                       method=tf.image.ResizeMethod.BILINEAR)
		return x_img_resized, ys

	# Processing the image filenames
	fs = os.listdir(path)
	csv_name = os.path.join(path, [it for it in fs if '.csv' in it][0])

	# Add one more column named "Train" to split the training set and validation set
	frame = pd.read_csv(csv_name)
	frame = frame.loc[frame['Train'] == 'F']
	print(' [*] {} images initialized as validation data'.format(frame['num_id'].count()))

	num_idx = frame['num_id'].values.astype(str).tolist()
	t_names = [item + '.jpg' for item in num_idx]
	file_names = [os.path.join(path, item) for item in t_names]
	labels = frame['Cloud_Cover'].values.tolist()
	t_labels = [list('F'.join(item.split('*'))) for item in labels]
	for it in range(len(t_labels)):
		t_labels[it] = list(map(lambda x: ord(x) - ord('A'), t_labels[it]))
	# Initialize as a tensorflow tensor object
	data = tf.data.Dataset.from_tensor_slices((tf.constant(file_names),
	                                           tf.constant(t_labels)))
	data = data.map(_parse_function)
	return data.batch(batch_size)


def init_loss(logits, labels, end_points=None, losstype='ordinal'):
	print(logits.get_shape(), labels.get_shape())
	if end_points is not None:
		# Definition of binary network for better classification of "*"
		# The network has only 3 layers, with the front-end being resnet_v1_152/block3
		# See the graph in tensorboard for more detailed information
		with tf.variable_scope('binary_classification_for_nodata'):
			conv_1 = slim.conv2d(end_points['resnet_v1_152/block4'], 64, [3, 3], scope='conv_1')
			conv_2 = slim.conv2d(conv_1, 1, [3, 3], scope='conv_2')
			reshaped = tf.reshape(conv_2, [FLAGS.batch_size*8, -1], name='reshaped')
			binary = slim.fully_connected(reshaped, 1, activation_fn=None, scope='fc_3')
			binary_labels = tf.reshape(tf.cast(tf.equal(labels, 5), tf.float32), [-1, 1])
			binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_labels,
		                                                        logits=binary)
			binary_loss = tf.reduce_mean(binary_loss, name='binary_loss')
	
	# Here we start our cross entropy loss definition
	if losstype == 'cross_entropy':
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
		                                                      logits=logits)
		return tf.reduce_mean(loss, name='loss') + binary_loss, binary
	elif losstype == 'ordinal':
		import math
		ks = [np.arange(1, 7).astype(np.float32)[None, :] \
		       for _ in range(FLAGS.batch_size * 8)]
		ks = np.concatenate(ks, axis=0)
		kfac = [[math.factorial(it) for it in range(1, 7)] for _ in range(FLAGS.batch_size * 8)]
		kfac = np.array(kfac, dtype=np.float32)
		k_vector = tf.constant(ks, name='k_vector')
		k_factor = tf.constant(kfac, name='k_factor')
		softmaxed = tf.nn.softmax(logits, axis=-1, name='softmax')
		log_exp = tf.log(softmaxed)
		poisson = k_vector * log_exp - logits - tf.log(k_factor)
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=poisson)
		return tf.reduce_mean(loss, name='loss') + binary_loss, binary
	else:
		raise NotImplementedError


class Confusion(object):
	'''
	 To compute the confusion matrix
	 The __str__ function is overrided for better visualization effect
	 Use the following command to install necessary packages
	 $ pip install tabulate
	'''
	def __init__(self, sess, pred_op, binary_op, label_op):
		self.matrix = np.zeros((6, 6), dtype=np.int64)
		counter = 0
		while True:
			try:
				pred, binary, ys = sess.run([pred_op, binary_op, label_op])
				print(pred, binary)
				for it in range(binary.shape[0]):
					if binary[it] > 0.5:
						self.matrix[5][ys[it]] += 1
					else:
						self.matrix[pred[it]][ys[it]] += 1
			except tf.errors.OutOfRangeError:
				print('Epoch reach the end...')
				break
			except tf.errors.InvalidArgumentError:
				print('An error of type tf.errors.InvalidArgumentError has been ignored...')
				continue
			counter += 1
			if counter % 200 == 5:
				print(self.matrix)
		print(' [*] Confusion matrix initialized!')

	def __str__(self):
		matrix = self.matrix.tolist()
		headers = ['A', 'B', 'C', 'D', 'E', '*']
		return tabulate(matrix, headers=headers, tablefmt='grid')


def main(_):
	reader = init_reader(FLAGS.data_dir)
	batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
	# param batch_xs: shape [-1, 512, 512, 3] type tf.float32
	# param batch_ys: shape [-1] type tf.int32
	off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
	off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
	x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 128, 256)\
		              	for hs, ws in zip(off_hs, off_ws)]
	batch_xs = tf.reshape(tf.concat(x_img_cuts, axis=0), [FLAGS.batch_size*8, 128, 256, 3])
	batch_ys = tf.reshape(batch_ys, [FLAGS.batch_size * 8])

	with slim.arg_scope(resnet_v1.resnet_arg_scope()):
		probs, end_points = resnet_v1.resnet_v1_152(batch_xs, num_classes=6, is_training=False)
		probs = tf.reshape(probs, [-1, 6])
		prediction = tf.argmax(probs, axis=-1)
		loss, binary = init_loss(probs, batch_ys, end_points, losstype=FLAGS.losstype)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	counter = get_counter(FLAGS.model_dir)

	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, counter, 
	                                           100, 0.98, staircase=True)
	if 'SGD' in FLAGS.optimizer:
		optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
		                                                                  global_step=tf.Variable(counter))
	elif 'Adam' in FLAGS.optimizer:
		optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(counter))
	else:
		raise NotImplementedError
	
	# Initialize all variables
	# Load the pretrained model and initialize the global counter
	sess.run(tf.global_variables_initializer())
	counter = load(sess, FLAGS.model_basedir)

	confusion_matrix = Confusion(sess, prediction, binary, batch_ys)
	print(confusion_matrix)



if __name__ == '__main__':
	tf.app.run()
