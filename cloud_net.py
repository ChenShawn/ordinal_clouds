import tensorflow as tf
import numpy as np
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from json_parser import TFNetworkParserTrain, TFNetworkParserTest

default_path = 'D:\\cloud\\mode_M2009\\'

def init_reader(path=default_path, batch_size=8, epoch=10, shuffle=True):
    def _parse_function(xs, ys):
        x_img_str = tf.read_file(xs)
        x_img_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(x_img_str),
                                                     tf.float32)
        x_img_resized = tf.image.resize_images(x_img_decoded, size=[512, 512],
                                               method=tf.image.ResizeMethod.BILINEAR)
        return x_img_resized, ys

    # Processing the image file names
    fs = os.listdir(path)
    csv_name = os.path.join(path, [it for it in fs if '.csv' in it][0])
    frame = pd.read_csv(csv_name)
    num_idx = frame['num_id'].values.astype(str).tolist()
    t_names = [item + '.jpg' for item in num_idx]
    file_names = [os.path.join(path, item) for item in t_names]
    # Processing the labels
    labels = frame['Cloud_Cover'].values.tolist()
    t_labels = [list('F'.join(item.split('*'))) for item in labels]
    for it in range(len(t_labels)):
        t_labels[it] = list(map(lambda x: ord(x) - ord('A'), t_labels[it]))

    # Initialize as the tensorflow tensor object
    data = tf.data.Dataset.from_tensor_slices((tf.constant(file_names),
                                               tf.constant(t_labels)))
    data = data.map(_parse_function)
    # return data.shuffle(buffer_size=1024).batch(batch_size).repeat(epoch)
    if shuffle:
        return data.shuffle(buffer_size=1024).batch(batch_size).repeat(epoch)
    else:
        return data.batch(batch_size)


class CloudNetworkTrain(TFNetworkParserTrain):
    def __init__(self, json_dir, learning_rate=4e-4,  batch_size=8,
                 model_dir='./ckpt/', pre_train=True):
        self.reader = init_reader(batch_size=batch_size, shuffle=True)
        batch_xs, batch_ys = self.reader.make_one_shot_iterator().get_next()
        off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
        off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
        x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 128, 256)\
                      for hs, ws in zip(off_hs, off_ws)]
        self.batch_xs = tf.reshape(tf.concat(x_img_cuts, axis=0),
                                   [batch_size * 8, 128, 256, 1])
        self.batch_ys = tf.reshape(batch_ys, [-1])
        super(CloudNetworkTrain, self).__init__({'xs': self.batch_xs, 'ys': self.batch_ys},
                                                json_dir, learning_rate, model_dir, pre_train)
        self.name = 'net'

    def train(self, steps, log_dir='./logs/', log_iter=40):
        writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        for it in range(steps):
            try:
                self.sess.run(self.optim)
                if it % log_iter == 20:
                    sum_str, loss, mAP = self.sess.run([self.summary, self.loss, self.mAP])
                    writer.add_summary(sum_str, self.counter)
                    words = ' --iteration {} --loss {} --mAP {}'.format(it, loss, mAP)
                    print(datetime.now(), words)
            except tf.errors.InvalidArgumentError:
                continue
            except:
                break
            self.counter += 1
        print('Training finish...Saving...')
        self.save()


class CloudNetworkTest(TFNetworkParserTest):
    def __init__(self, json_dir, batch_size=8, model_dir='./ckpt/', pre_train=True):
        self.reader = init_reader(batch_size=batch_size, shuffle=False)
        batch_xs, batch_ys = self.reader.make_one_shot_iterator().get_next()
        off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
        off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
        x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 128, 256)\
                      for hs, ws in zip(off_hs, off_ws)]
        self.batch_xs = tf.reshape(tf.concat(x_img_cuts, axis=0),
                                   [batch_size * 8, 128, 256, 1])
        self.batch_ys = tf.reshape(batch_ys, [-1])
        super(CloudNetworkTest, self).__init__({'xs': self.batch_xs, 'ys': self.batch_ys},
                                               json_dir, model_dir, pre_train)
        self.model_dir = model_dir

    def evaluate(self, *args, **kwargs):
        accuracy = 0.0
        while True:
            try:
                accuracy, _ = self.sess.run([self.accuracy, self.update_acc])
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                continue
            except tf.errors.NotFoundError:
                break
        return accuracy



if __name__ == '__main__':
    # reader = init_reader(shuffle=False)
    # batch_xs, batch_ys = reader.make_one_shot_iterator().get_next()
    # off_ws = [0, 0, 0, 0, 256, 256, 256, 256]
    # off_hs = [0, 128, 256, 384, 0, 128, 256, 384]
    # x_img_cuts = [tf.image.crop_to_bounding_box(batch_xs, hs, ws, 128, 256) \
    #               for hs, ws in zip(off_hs, off_ws)]
    # batch_xs = tf.concat(x_img_cuts, axis=0)
    # batch_ys = tf.reshape(batch_ys, [-1])
    #
    # with tf.Session() as sess:
    #     x, y = sess.run([batch_xs, batch_ys])
    #
    #     print(x.shape, y, y.shape)
    #     print(batch_xs, batch_ys)

    TRAIN_STEPS = int(sys.argv[1])
    cloud = CloudNetworkTrain('./json2tf.json', pre_train=True)
    cloud.train(TRAIN_STEPS)