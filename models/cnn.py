import tensorflow as tf
slim = tf.contrib.slim

class SimpleLeNet(object):
    def __init__(self):
        self.n_hidden_1 = 2048
        self.n_hidden_2 = 1024
        self.n_input = 100*100
        self.n_classes = 120
        self.learning_rate = 0.001
        self.optimizer = None

        self.image_batch = tf.placeholder(tf.float32, shape=[None, 100,100,1])
        self.label_batch = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.is_training = tf.placeholder(tf.bool)

        self.net = slim.conv2d(self.image_batch, 8, [5,5], scope='conv1',activation_fn=tf.nn.relu)
        self.net = slim.max_pool2d(self.net, [2,2], scope='pool1')

        self.net = slim.conv2d(self.net, 16, [5,5], scope='conv2', activation_fn=tf.nn.relu)
        self.net = slim.max_pool2d(self.net, [2,2], scope='pool2')

        self.net = slim.conv2d(self.net, 32, [5,5], scope='conv3', activation_fn=tf.nn.relu)
        self.net = slim.max_pool2d(self.net, [2,2], scope='pool3')

        self.net = slim.conv2d(self.net, 64, [5,5], scope='conv4', activation_fn=tf.nn.relu)
        self.net = slim.max_pool2d(self.net, [2,2], scope='pool4')

        self.net = slim.flatten(self.net, scope='flatten3')
        self.net = slim.fully_connected(self.net, 512, scope='fc4', activation_fn=tf.nn.sigmoid)
        self.net = slim.batch_norm(self.net,activation_fn=None)


        self.net = slim.fully_connected(self.net, self.n_classes, activation_fn=None, scope='fc5')

        self.pred = self.net

        slim.losses.softmax_cross_entropy(
            self.pred,
            self.label_batch)

        self.cost = slim.losses.get_total_loss()
        tf.summary.scalar('loss', self.cost)

        self.optimizer = tf.train.AdamOptimizer(0.01, 0.9)

        # create train op
        self.train_op = self.optimizer.minimize(self.cost)
