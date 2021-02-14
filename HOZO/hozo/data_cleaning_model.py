import tensorflow as tf
import rfho as rf
import numpy as np
import os

class data_cleaning_mode:

    def __init__(self, lmd, **kw):

        self.lmd = lmd
        self.T = kw['T']
        self.lr = kw['lr']
        self.times = kw['times']
        # TODO:num_gpus -> gpu_list
        num_gpus = kw['num_gpus']
        
        # create & build graph
        self.graph = tf.Graph()
        self.build_graph()

        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # This will result an unknown error!
        # local_device_protos = device_lib.list_local_devices()
        # num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        # print(num_gpus)

        # Don't let gpu 0 run too much by using +2.
        gpu_id = (os.getpid() + 2) % num_gpus
        config.gpu_options.visible_device_list= str(gpu_id)
        self.sess = tf.Session(config=config,graph=self.graph)

    def build_graph(self):
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')

            W = tf.Variable(tf.zeros([784, 10],tf.float32), name='W')
            b = tf.Variable(tf.zeros([10],tf.float32), name='b')

            self.out = tf.matmul(self.x, W) + b

            ce = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.out)
            
            correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
            
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.error = tf.reduce_mean(ce)
            full_lmd = np.tile(self.lmd, self.times)
            self.weighted_error = tf.reduce_mean(tf.sigmoid(full_lmd) * ce)
            opt = tf.train.GradientDescentOptimizer(self.lr)
            self.ts = opt.minimize(self.weighted_error)

    def train_valid(self, **kw):
        data = kw['data']
        train_s = data.train.create_supplier(self.x, self.y)
        valid_s = data.validation.create_supplier(self.x, self.y)
        with self.sess:
            tf.global_variables_initializer().run()
            for t_idx in range(self.T):
                self.ts.run(feed_dict=train_s())

            # train_error = self.weighted_error.eval(feed_dict=train_s())
            valid_error = self.error.eval(feed_dict=valid_s())

            # return valid_error, train_error, test_error
            return valid_error

    def __del__(self):
        # print('RELEASE RESOURCE!')
        self.sess.close()
        del self.lmd
        del self.T
        del self.lr
        del self.graph
        del self.sess
