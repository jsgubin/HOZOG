import numpy as np
from scipy import sparse
from sklearn import linear_model
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import time
import hozo


class DataHyperCleaningBO(linear_model.base.BaseEstimator,
                           linear_model.base.LinearClassifierMixin):

    def __init__(
                 self, data, T=2000, lr=.1, loss_function='ce'):
        self.data = data
        self.T = T
        self.lr = lr

        if(loss_function == 'ce'):
            self.cost_func = self.cost_func_ce
        elif(loss_function == '01'):
            self.cost_func = self.cost_func_01
        else:
            print('Please specify a valid loss function')

    def black_box_function(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
        np.random.seed(0)
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)
        tmp_lambdak=np.ones(self.data.train.num_examples)*0.0
        part_size = int(np.ceil(self.data.train.num_examples / 20))
        for p_idx in range(20):
            if p_idx == 19:
                true_idx = p_idx + 1
                # print('true_idx',true_idx)
                tmp_lambdak[part_size*p_idx:]=np.ones(self.data.train.num_examples-19*part_size)*eval('x'+str(true_idx))
            else:
                # print(part_size*p_idx)
                # print(part_size*(p_idx+1)-1)
                # print(tmp_part_lambdak[p_idx])
                true_idx = p_idx + 1
                # print(eval('x'+str(true_idx)))
                tmp_lambdak[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size)*eval('x'+str(true_idx))   
        saver=None
        # print(tmp_lambdak)
        f = self.cost_func(saver, model, y, self.data, self.T, self.lr, tmp_lambdak, name='Test') 
        return f 

    def black_box_function100(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
        x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
        x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, \
        x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, \
        x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, \
        x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, \
        x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, \
        x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, \
        x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, \
        x91, x92, x93, x94, x95, x96, x97, x98, x99, x100):
        np.random.seed(0)
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        print('Call function')
        model = hozo.LinearModel(x, 28 * 28, 10)
        tmp_lambdak=np.ones(self.data.train.num_examples)*0.0
        part_size = int(np.ceil(self.data.train.num_examples / 100))
        for p_idx in range(100):
            if p_idx == 99:
                true_idx = p_idx + 1
                tmp_lambdak[part_size*p_idx:]=np.ones(self.data.train.num_examples-99*part_size)*eval('x'+str(true_idx))
            else:
                # print(part_size*p_idx)
                # print(part_size*(p_idx+1)-1)
                # print(tmp_part_lambdak[p_idx])
                true_idx = p_idx + 1
                # print(eval('x'+str(true_idx)))
                tmp_lambdak[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size)*eval('x'+str(true_idx))   
        saver=None
        # print(tmp_lambdak)
        f = self.cost_func(saver, model, y, self.data, self.T, self.lr, tmp_lambdak, name='Test') 
        return f 


    def cost_func_ce(self, saver, model, y, data, T, lr, lmd=None, name=None):
        # TODO other optimizers?
        """
        BASELINE EXECUTION (valid also for oracle and final training,
        with optimized values of lambda)

        :param saver: `Saver` object (can be None)
        :param name: optional name for the saver
        :param data: `Datasets` object
        :param T: number of iterations
        :param lmd: weights for the examples, if None sets to 1.
        :param model: a model (should comply with `rf.Network`)
        :param y: placeholder for output
        :param lr: learning rate
        :return:
        """
        # if saver: saver.save_setting(vars(), append_string=name)
        x = model.inp[0]

        # def _train_and_valid_s():
        #     return {x: np.vstack((data.train.data, data.validation.data)),
        #             y: np.vstack((data.train.target, data.validation.target))}

        train_s = data.train.create_supplier(x, y)
        valid_s = data.validation.create_supplier(x, y)

        error2 = tf.reduce_mean(lmd * hozo.cross_entropy_loss(y, model.out))
        correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
        error = tf.reduce_mean(hozo.cross_entropy_loss(y, model.out))

        opt = tf.train.GradientDescentOptimizer(lr)
        ts1 = opt.minimize(error2, var_list=model.var_list)

        # saver related
        if saver:
            saver.clear_items()
            saver.add_items(
                'Test Accuracy', accuracy2, tst_s,
            )

        with tf.Session(config=hozo.CONFIG_GPU_GROWTH).as_default():
            tf.variables_initializer(model.var_list).run()
            for _ in range(T):
                ts1.run(feed_dict=train_s())
            if saver: saver.save(name)
            baseline_test_accuracy = accuracy2.eval(feed_dict=valid_s())
            test_error = error.eval(feed_dict=valid_s())
            # print('baseline_test_accuracy',baseline_test_accuracy)
            # return 1-baseline_test_accuracy
            return -test_error

    def cost_func_01(self, saver, model, y, data, T, lr, lmd=None, name=None):
        # TODO other optimizers?
        """
        BASELINE EXECUTION (valid also for oracle and final training,
        with optimized values of lambda)

        :param saver: `Saver` object (can be None)
        :param name: optional name for the saver
        :param data: `Datasets` object
        :param T: number of iterations
        :param lmd: weights for the examples, if None sets to 1.
        :param model: a model (should comply with `rf.Network`)
        :param y: placeholder for output
        :param lr: learning rate
        :return:
        """
        # if saver: saver.save_setting(vars(), append_string=name)
        x = model.inp[0]

        # def _train_and_valid_s():
        #     return {x: np.vstack((data.train.data, data.validation.data)),
        #             y: np.vstack((data.train.target, data.validation.target))}

        train_s = data.train.create_supplier(x, y)
        valid_s = data.validation.create_supplier(x, y)

        error2 = tf.reduce_mean(lmd * hozo.cross_entropy_loss(y, model.out))
        correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
        error = tf.reduce_mean(hozo.cross_entropy_loss(y, model.out))

        opt = tf.train.GradientDescentOptimizer(lr)
        ts1 = opt.minimize(error2, var_list=model.var_list)

        # saver related
        if saver:
            saver.clear_items()
            saver.add_items(
                'Test Accuracy', accuracy2, tst_s,
            )

        with tf.Session(config=hozo.CONFIG_GPU_GROWTH).as_default():
            tf.variables_initializer(model.var_list).run()
            for _ in range(T):
                ts1.run(feed_dict=train_s())
            if saver: saver.save(name)
            baseline_test_accuracy = accuracy2.eval(feed_dict=valid_s())
            test_error = error.eval(feed_dict=valid_s())
            # print('baseline_test_accuracy',baseline_test_accuracy)
            return baseline_test_accuracy-1
            # return -test_error