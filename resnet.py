import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2, resnet_v1
import numpy as np
import pandas as pd
import os, sys
from datetime import datetime

# TODO: output the results to file "results.csv" and compute the confusion matrix

flags = tf.app.flags
ERROR_FLAG = 0

flags.DEFINE_string('pretrain_dir', './checkpoints/resnet_v1_152.ckpt', '')
flags.DEFINE_string('model_dir', './checkpoints/models/resnet_ordinal.model', '')
flags.DEFINE_string('data_dir', '../../datasets/cloud/mode_2004/', '')
flags.DEFINE_string('model_basedir', './checkpoints/models/', '')
flags.DEFINE_string('logdir', './logs/', '')
flags.DEFINE_string('optimizer', 'SGD', 'Either Adam or SGD')
flags.DEFINE_string('losstype', 'cross_entropy', 'Either ordinal or cross_entropy')
flags.DEFINE_boolean('is_training', True, 'Train or evaluate?')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('loops', 40000, 'Number of iteration in training')
flags.DEFINE_float('learning_rate', 8e-3, 'Initial learning rate')
flags.DEFINE_boolean('pretrained', False, 'Whether using the pretrained model given by TensorFlow')

FLAGS = flags.FLAGS


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
		print(" [*] Failed to find a checkpoint")
		return ERROR_FLAG


def get_uninitialized_vars(sess):
	# This method is too slow here for ResNet 
	# mainly due to the large number of layers and variables
	train_vars = tf.trainable_variables()
	res = list()
	for var in train_vars:
		try:
			sess.run(var)
			print('Going through................')
		except tf.errors.FailedPreconditionError:
			res.append(var)
			continue
	return res


def train(sess, optim, loss, summaries, loop=20000, 
          train_vars=None, counter=0, logiter=50):
	writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
	global_counter = counter
	for it in range(loop):
		try:
			sess.run(optim)
			if it % logiter == 2:
				sum_str, lossval = sess.run([summaries, loss])
				writer.add_summary(sum_str, global_counter)
				words = ' --iteration {} --loss {}'.format(it, lossval)
				print(str(datetime.now()) + words)
		except tf.errors.InvalidArgumentError:
			print('An error of type tf.errors.InvalidArgumentError has been ignored...')
			continue
		except tf.errors.OutOfRangeError:
			print('Epoch reach the end...')
			break
		global_counter += 1
	print('Training finish...Trying to save the model...')
	save(sess, FLAGS.model_dir, global_counter)


def evaluate(sess, accuracy, acc_update):
	accval = 0.0; cnt = 0
	sess.run(tf.local_variables_initializer())
	while cnt < 500000:
		try:
			accval, _ = sess.run([accuracy, acc_update])
		except tf.errors.InvalidArgumentError:
			print('An error of type tf.errors.InvalidArgumentError has been ignored...')
			continue
		except tf.errors.OutOfRangeError:
			break
		except tf.errors.NotFoundError:
			break
		cnt += 1
	return accuracy


def init_reader(path=FLAGS.data_dir, batch_size=8, epoch=20, is_training=True):
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
	if is_training:
		frame = pd.read_csv(csv_name)
		frame = frame.loc[frame['Train'] == 'T']
		print(' [*] {} images initialized as training data'.format(frame['num_id'].count()))
	else:
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
	if is_training:
		return data.shuffle(buffer_size=1024).batch(batch_size).repeat(epoch)
	else:
		return data.batch(batch_size)


def init_loss(logits, labels, end_points=None, losstype='ordinal'):
	if end_points is not None:
		# Definition of binary network for better classification of "*"
		# The network has only 3 layers, with the front-end being resnet_v1_152/block4
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
	# Note that the "nodata" class "*" is assigned with 0 loss
	if losstype == 'cross_entropy':
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
		                                                      logits=logits)
		mask = 1.0 - tf.cast(tf.equal(labels, 5), tf.float32)
		return tf.reduce_mean(mask * loss, name='loss') + binary_loss, binary
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
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
		                                                      logits=poisson)
		mask = 1.0 - tf.cast(tf.equal(labels, 5), tf.float32)
		return tf.reduce_mean(loss * mask, name='loss') + binary_loss, binary
	else:
		raise NotImplementedError


def main(_):
	reader = init_reader(FLAGS.data_dir, batch_size=FLAGS.batch_size)
	batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
	# param batch_xs: shape [-1, 512, 512, 3] type tf.float32
	# param batch_ys: shape [-1] type tf.int32
	off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
	off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
	x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 128, 256)\
	              	for hs, ws in zip(off_hs, off_ws)]
	batch_xs = tf.reshape(tf.concat(x_img_cuts, axis=0), [FLAGS.batch_size*8, 128, 256, 3])
	batch_ys = tf.reshape(batch_ys, [FLAGS.batch_size * 8])

	if FLAGS.is_training:
		with slim.arg_scope(resnet_v1.resnet_arg_scope()):
			logits, end_points = resnet_v1.resnet_v1_152(batch_xs, num_classes=6, 
																									 is_training=True)
			logits = tf.reshape(logits, [-1, 6], name='logits_2d')
			prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
			mAP = tf.reduce_mean(tf.cast(tf.equal(prediction, batch_ys), 
			                             dtype=tf.float32))
			loss, _ = init_loss(logits, batch_ys, end_points=end_points, losstype=FLAGS.losstype)
			mAP_sum = tf.summary.scalar('mAP', mAP)
			loss_sum = tf.summary.scalar('loss', loss)
			summaries = tf.summary.merge([mAP_sum, loss_sum])
			
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.InteractiveSession(config=config)
		counter = get_counter(FLAGS.model_dir)

		# Exponential decay learning rate and optimizer configurations
		learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, counter, 
		                                           100, 0.98, staircase=True)
		if 'SGD' in FLAGS.optimizer:
			optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
			                                                                  global_step=tf.Variable(counter))
		elif 'Adam' in FLAGS.optimizer:
			optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(counter))
		else:
			optim = None
			raise NotImplementedError
		sess.run(tf.global_variables_initializer())

		if FLAGS.pretrained:
			# Load the pretrained model given by TensorFlow official
			exclusions = ['resnet_v1_152/logits', 'predictions']
			resnet_except_logits = slim.get_variables_to_restore(exclude=exclusions)
			init_fn = slim.assign_from_checkpoint_fn(FLAGS.pretrain_dir, resnet_except_logits,
			                                         ignore_missing_vars=True)
			init_fn(sess)
			print('Model successfully loaded')
		else:
			# Load the model trained by ourselves
			counter = load(sess, FLAGS.model_basedir)

		# Ready to train
		train(sess, optim, loss, summaries, FLAGS.loops, counter=counter)
		print('Training finished')
	else:
		with slim.arg_scope(resnet_v1.resnet_arg_scope()):
			probs, end_points = resnet_v1.resnet_v1_152(batch_xs, num_classes=6,
																									is_training=False)
			prediction = tf.argmax(tf.reshape(probs, [-1, 6]), axis=-1)
			accuracy, update_acc = tf.metrics.accuracy(batch_ys, arg_ys)
			_, binary = init_loss(probs, batch_ys, end_points, loss_type=FLAGS.losstype)

		# TODO: Consider the binary classification... 
		# which is even more tricky to be implemented...
		print('The model accuracy is {}'.format(evaluate(sess, accuracy, update_acc)))



if __name__ == '__main__':
	tf.app.run()

