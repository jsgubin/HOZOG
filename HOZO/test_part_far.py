#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import rfho as rf
from rfho.datasets import load_mnist
import far_ho as far
import hozo
import time
import cvxopt
from cvxopt import spmatrix, matrix, sparse

def load_std_data(folder=None, limit_ex=20000, sets_proportions=(.1, .1),
                  noise=.5):
    """
    If necessary, set the seed before calling this funciton

    :param folder:
    :param limit_ex:
    :param sets_proportions:
    :param noise: noise level
    :return:
    """

    # noinspection PyUnusedLocal
    def _limit_examples(_x, _y, _d, _k):
        return _k < limit_ex

    mnist = load_mnist(folder, partitions=sets_proportions, filters=_limit_examples)

    # ADD NOISE TO LABELS
    np.random.seed(0)
    noisy_indices = np.sort(np.random.permutation(mnist.train.num_examples)[
                            :int(mnist.train.num_examples * noise)])
    clean_indices = list(set(range(mnist.train.num_examples)) - set(noisy_indices))

    for fl in noisy_indices:  # randomly change targets
        new_lab = np.random.choice(list(set(range(10))
                                        - {np.argmax(mnist.train.target[fl])}))
        mnist.train.target[fl] = np.zeros(10, dtype='float32')
        mnist.train.target[fl][new_lab] = 1.

    info_dict = {
        'noisy examples': noisy_indices,
        'clean examples': clean_indices,
        'N noisy': len(noisy_indices),
        'N clean': len(clean_indices)
    }

    mnist.train.info = info_dict  # save some useful stuff

    return mnist

def data_hyper_cleaner(saver, data, T, lr, R, partition=10,
                       hyper_lr=0.05, hyper_iterations=50000, debug=False):
    start_time = time.time()
    name = '_newP_T=%d_lr=%f'%(T,lr)
    train_time = 0
    time_list = []
    v_error_list = []
    tr_error_list = []
    t_error_list = []
    if debug:
        lmd_list = []
        grad_list = []

    x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
    # model = hozo.LinearModel(x, 28 * 28, 10)
    
    # _, out = rf.vectorize_model(model.var_list, model.out, augment=0)

    # logits = model.out
    W = tf.Variable(tf.zeros([784, 10],tf.float32), name='W')
    b = tf.Variable(tf.zeros([10],tf.float32), name='b')

    logits = tf.matmul(x, W) + b
    

    lambdas = far.get_hyperparameter('lambdas', 0.1*tf.ones(partition), dtype='float32')
    ex_lambdas = tf.tile(lambdas,[int(data.train.num_examples/partition)])
    print(lambdas)
    print(ex_lambdas)

    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    L = tf.reduce_mean(tf.sigmoid(ex_lambdas)*ce)
    E = tf.reduce_mean(ce)

    inner_optimizer = far.GradientDescentOptimizer(0.1)
    # outer_optimizer = tf.train.AdamOptimizer(learning_rate=hyper_lr)
    outer_optimizer = tf.train.GradientDescentOptimizer(learning_rate=hyper_lr)
    hyper_step = far.HyperOptimizer().minimize(E, outer_optimizer, L, inner_optimizer)

    # projection
    grad_hyper = tf.placeholder(tf.float32)
    lmd_assign = lambdas.assign(grad_hyper)

    _project = _get_projector(R=R, N_ex=partition)

    def projection(lmd):
        lmd[lmd < 0] = 0
        lmd[lmd > 1] = 1
        return lmd

    def project():
        pt = lambdas.eval()
        # _resx = projection(pt)
        _resx = _project(pt)
        lmd_assign.eval(feed_dict={grad_hyper: _resx})

    # suppliers
    tr_s = data.train.create_supplier(x, y)
    val_s = data.validation.create_supplier(x, y)
    tst_s = data.test.create_supplier(x, y)


    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():  
        tf.global_variables_initializer().run()
        if debug:
            lmd_list.append(lambdas.eval())
        for hyt in range(hyper_iterations):
            hyper_step(T,
               inner_objective_feed_dicts=tr_s,
               outer_objective_feed_dicts=val_s)
            # project()
            if debug:
                print(lambdas.eval())
                lmd_list.append(lambdas.eval())
                grad_list.append(far.hypergradients()[-1].eval())
                np.savetxt("./experiments/lmd_RHO%s.txt"%name,lmd_list)
                np.savetxt("./experiments/grad_RHO%s.txt"%name,grad_list)
            if (hyt%20==0):
                train_time = train_time + (time.time()-start_time)
                v_error=E.eval(feed_dict=val_s())
                # tr_error=L.eval(feed_dict=tr_s())
                # t_error=E.eval(feed_dict=tst_s())
                time_list.append(train_time)
                v_error_list.append(v_error)
                # tr_error_list.append(tr_error)
                # t_error_list.append(t_error)
                print('----Iter:',hyt)
                print('----Error:',v_error_list)
                # print('----Train Error:',tr_error_list)
                # print('----Test Error:',t_error_list)
                print('----Time:',time_list)
                start_time = time.time()
        train_time = train_time + (time.time()-start_time)
        return lambdas.eval()


def _get_projector(R, N_ex):  # !
    # Projection
    dim = N_ex
    P = spmatrix(1, range(dim), range(dim))
    glast = matrix(np.ones((1, dim)))
    G = sparse([-P, P, glast])
    h1 = np.zeros(dim)
    h2 = np.ones(dim)
    h = matrix(np.concatenate([h1, h2, [R]]))

    def _project(pt):
        print('start projection')
        # pt = gamma.eval()
        q = matrix(- np.array(pt, dtype=np.float64))
        # if np.linalg.norm(pt, ord=1) < R:
        #    return
        _res = cvxopt.solvers.qp(P, q, G, h, initvals=q)
        _resx = np.array(_res['x'], dtype=np.float32)[:, 0]
        # gamma_assign.eval(feed_dict={grad_hyper: _resx})
        return _resx

    return _project

def main(saver=None, run_baseline=True, run_oracle=True, run_optimization=True,
         run_rho=True, T=2000, lr=.1):

    # Close Tensorflow Log
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    np.random.seed(0)

    # sets_proportions: list of fractions for training set and test set
    data = load_std_data(sets_proportions=(.5, .45),limit_ex=4000)
    print(data.train.num_examples)
    # print(data.train.info['noisy examples'])
    # print(data.train.info['clean examples'])

    if run_rho:
        partition = data.train.num_examples
        R=partition
        normalized_lr = lr
        lmd = data_hyper_cleaner(saver, data, T, lr=normalized_lr, R=R, partition=partition,
                                         hyper_lr=40, hyper_iterations=2000, debug=False)


if __name__ == '__main__':
    main(run_baseline=False, run_oracle=False, run_optimization=False,
         run_rho=True,T=2000)