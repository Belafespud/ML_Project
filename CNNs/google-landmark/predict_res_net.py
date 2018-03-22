import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import train
from nets import nets_factory, resnet_v2
from utils.custom_preprocessing import preprocessing_factory
from utils.dataset import GoogleDataset

slim = tf.contrib.slim


# def get_model():
    
#     exclusions = []
#     variables_to_restore = []
    
#     for var in slim.get_model_variables():
#         excluded = False
#         for exclusion in exclusions:
#             if var.op.name.startswith(exclusion):
#                 excluded = True
#                 break
#         if not excluded:
#             variables_to_restore.append(var)

#     if tf.gfile.IsDirectory('output/resnet_full_next/'):
#         checkpoint_path = tf.train.latest_checkpoint('output/resnet_full_next/')
#     else:
#         checkpoint_path = 'output/resnet_full_next/'

#     tf.logging.info('Fine-tuning from %s' % 'output/resnet_full_next/')

#     return slim.assign_from_checkpoint_fn(
#         checkpoint_path,
#         variables_to_restore,
#         ignore_missing_vars=True)



dataset = GoogleDataset('data/')
test_preprocessing_fn = preprocessing_factory('crop_and_resize',
                                               output_height=180,
                                               output_width=180,
                                               is_training=True)

test_iterator = dataset.get_test_iterator(epochs=-1, batch_size=50,
                                                    preprocessing_function=test_preprocessing_fn, shuffle=True)
images, _ = test_iterator.get_next()

variables_to_restore = []
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    
    print(slim.get_model_variables())

    for var in slim.get_model_variables():
        variables_to_restore.append(var)

    checkpoint_path = tf.train.latest_checkpoint('output/resnet_full_next/')

    my_model = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=True)

    print(my_model)
    logits, _ = my_model(images)
    predictions = tf.nn.softmax(logits)
    predicted_labels = tf.to_int32(tf.argmax(predictions, axis=1))


with tf.Session() as sess:
    sess.run(predicted_labels)