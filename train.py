
# Copyright 2017 TEMALAB. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import models.cnn.SimpleAlexNet

FLAGS = tf.app.flags.FLAGS

def train():
    input_image = tf.placeholder([batch_size], 0,0,0), tf.float32)
    net = SimpleAlexNet()
    predict_logit = net.output
    # train_op = optimizer.minmize(net.loss)

    with tf.Session() as session:
        pass


def main(_):
    pass


if __name__ == "__main__":
  tf.app.run()
