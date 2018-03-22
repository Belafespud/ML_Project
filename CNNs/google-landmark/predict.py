import numpy as np
import os
import tensorflow as tf
import urllib2
import csv
slim = tf.contrib.slim

from nets import resnet_v2
import utils
from nets import resnet_utils
from utils.custom_preprocessing import crop_and_resize

checkpoints_dir = 'output/resnet_full_next'
image_size = 180
outfile = 'answers.csv'
string = ''
string += 'id,landmarks\n'

    
for filename in os.listdir('data/test'):
    image = os.path.join('data/test', filename)
    f = open(image)
    image_string = f.read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = crop_and_resize(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_50(processed_images, num_classes=14951, is_training=False)

        probabilities = tf.nn.softmax(logits)
        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'model.ckpt-31909'), slim.get_model_variables('resnet_v2_50'))

    with tf.Session() as sess:
        init_fn(sess)
        np_image, network_input, probabilities = sess.run([image, processed_image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key = lambda x : x[1])]

        index = sorted_inds[0]
        string += '%s,%i %0.2f\n' % (filename[:-4], index, probabilities[index])

    res = slim.get_model_variables()
    tf.reset_default_graph()
    
with open(outfile, 'w') as outfp:
    outfp.write(string)