#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from rfho.datasets import load_mnist
import hozo
import time
from bayes_opt import BayesianOptimization
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

    np.savetxt("./experiments/noisy_indices.txt",noisy_indices)

    info_dict = {
        'noisy examples': noisy_indices,
        'clean examples': clean_indices,
        'N noisy': len(noisy_indices),
        'N clean': len(clean_indices)
    }

    mnist.train.info = info_dict  # save some useful stuff

    return mnist


def baseline(saver, model, y, data, T, lr, lmd=None, name='baseline'):
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
    train_and_valid = rf.datasets.Dataset.stack(data.train, data.validation)
    train_and_valid_s = train_and_valid.create_supplier(x, y)

    tst_s = data.test.create_supplier(x, y)

    if lmd is None: lmd = np.ones(train_and_valid.num_examples, dtype='float32')

    error2 = tf.reduce_mean(lmd * hozo.cross_entropy_loss(y, model.out))
    correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

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
            ts1.run(feed_dict=train_and_valid_s())
        if saver: saver.save(name)
        baseline_test_accuracy = accuracy2.eval(feed_dict=tst_s())
        return baseline_test_accuracy



def main(saver=None, run_baseline=True, run_oracle=True, run_optimization=True,
         run_bayesian=True, T=2000, lr=.1):
    # TODO ...
    """
    This method should replicate ICML experiment....
    
    :param R_search_space: 
    :param saver: 
    :param run_baseline: 
    :param run_oracle: 
    :param run_optimization: 
    :param T: 
    :param lr: 
    :return: 
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Ban gpus
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    np.random.seed(0)
    data = load_std_data(sets_proportions=(.005, .5),limit_ex=4000)
    # data.train = hozo.datasets.Dataset.stack(data.train, data.train)
    # data.train = hozo.datasets.Dataset.stack(data.train, data.train)
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = hozo.LinearModel(x, 28 * 28, 10)


    if run_baseline:
        # baseline_lrate = args.lr * args.p_train * args.N_all_ / N_unflipped  # better lr of optimizer i'd say
        print(baseline(saver, model, y, data, T, lr, name='Baseline'))  # could run for more iterations...

    if run_oracle:
        lmd_oracle = np.ones(data.train.num_examples + data.validation.num_examples, dtype='float32')
        lmd_oracle[data.train.info['noisy examples']] = 0.
        # for ind in data.train.info['noisy examples']:
        #     lmd_oracle[ind] = 0.
        print(baseline(saver, model, y, data, T, lr, lmd=lmd_oracle, name='Oracle'))

    if run_optimization:

        partition = data.train.num_examples
        part_lmd=np.ones(partition, dtype='float32')*0.1
        # part_lmd=np.ones(partition, dtype='float32')*0.1
        # part_lmd=np.zeros(partition, dtype='float32')
        lambda0= np.tile(part_lmd, int(data.train.num_examples/partition))
        # lambda0 = np.zeros(data.train.num_examples, dtype='float32')
 #        lambda0 =np.array([ 0.01385134,  0.01306581, -0.0021766,   0.01933503, -0.00336224,  0.00286622,
 # -0.00049531, -0.00241852,  0.01417598,  0.0090969,   0.00889277,  0.01099149,
 # -0.00387235, -0.01710998, -0.00419022,  0.00914826,  0.01330889, -0.00560561,
 #  0.01701175,  0.00285818],dtype='float32')
        # lambda0=np.random.random(data.train.num_examples)*0.1
        # print(lambda0)
        data_hyper_clean = hozo.DataHyperCleaning(lambda0=lambda0, 
            max_iter=10000, 
            eta=5,
            q=5, 
            mu=1e-3, 
            loss_function='ce')

        def projection(lmd):
            lmd[lmd < 0] = 0
            lmd[lmd > 1] = 1
            return lmd
        

        # _project = _get_projector(R=partition, N_ex=partition)
        # data_hyper_clean.fit(data, saver, T, lr, projection=projection)
        data_hyper_clean.fit_partition(data=data, 
            part_lmd=part_lmd, 
            saver=saver, 
            T=T, 
            lr=lr, 
            projection=projection,
            name='float32_fast')
        # data_hyper_clean.fit_partition_coordinate \
        # (data=data, part_lmd=part_lmd, saver=saver, T=T, lr=lr, cd_size=500, projection=projection)
        # data_hyper_clean.fit_partition_NAG(data, part_lmd, saver, T, lr, projection=projection, discount=0.1)
        print(data_hyper_clean.losses)
        print(data_hyper_clean.times)
        np.savetxt("./experiments/losses.txt",data_hyper_clean.losses)
        np.savetxt("./experiments/times.txt",data_hyper_clean.times)
        lmd_optimized = np.ones(data.train.num_examples + data.validation.num_examples, dtype='float32')
        lmd_optimized *= np.max(data_hyper_clean.lambdak)  # there's no reason to overweight the validation examples...
        lmd_optimized[:data.train.num_examples] = data_hyper_clean.lambdak
        print(baseline(saver, model, y, data, T, lr, lmd=lmd_optimized, name='Final Round'))
    
    if run_bayesian:
        data_hyper_clean_BO = hozo.DataHyperCleaningBO(data=data,loss_function='01')
        bo_start = time.time()
        pbounds = {'x1': (0, 1), 'x2': (0, 1),'x3': (0, 1),'x4': (0, 1),'x5': (0, 1),'x6': (0, 1),'x7': (0, 1),'x8': (0, 1),'x9': (0, 1),'x10': (0, 1),\
        'x11': (0, 1),'x12': (0, 1),'x13': (0, 1),'x14': (0, 1),'x15': (0, 1),'x16': (0, 1),'x17': (0, 1),'x18': (0, 1),'x19': (0, 1),'x20': (0, 1)}
        
        # pbounds = {'x1': (0, 1), 'x2': (0, 1),'x3': (0, 1),'x4': (0, 1),'x5': (0, 1),\
        # 'x6': (0, 1),'x7': (0, 1),'x8': (0, 1),'x9': (0, 1),'x10': (0, 1),\
        # 'x11': (0, 1),'x12': (0, 1),'x13': (0, 1),'x14': (0, 1),'x15': (0, 1),\
        # 'x16': (0, 1),'x17': (0, 1),'x18': (0, 1),'x19': (0, 1),'x20': (0, 1),\
        # 'x21': (0, 1),'x22': (0, 1),'x23': (0, 1),'x24': (0, 1),'x25': (0, 1),\
        # 'x26': (0, 1),'x27': (0, 1),'x28': (0, 1),'x29': (0, 1),'x30': (0, 1),\
        # 'x31': (0, 1),'x32': (0, 1),'x33': (0, 1),'x34': (0, 1),'x35': (0, 1),\
        # 'x36': (0, 1),'x37': (0, 1),'x38': (0, 1),'x39': (0, 1),'x40': (0, 1),\
        # 'x41': (0, 1),'x42': (0, 1),'x43': (0, 1),'x44': (0, 1),'x45': (0, 1),\
        # 'x46': (0, 1),'x47': (0, 1),'x48': (0, 1),'x49': (0, 1),'x50': (0, 1),\
        # 'x51': (0, 1),'x52': (0, 1),'x53': (0, 1),'x54': (0, 1),'x55': (0, 1),\
        # 'x56': (0, 1),'x57': (0, 1),'x58': (0, 1),'x59': (0, 1),'x60': (0, 1),\
        # 'x61': (0, 1),'x62': (0, 1),'x63': (0, 1),'x64': (0, 1),'x65': (0, 1),\
        # 'x66': (0, 1),'x67': (0, 1),'x68': (0, 1),'x69': (0, 1),'x70': (0, 1),\
        # 'x71': (0, 1),'x72': (0, 1),'x73': (0, 1),'x74': (0, 1),'x75': (0, 1),\
        # 'x76': (0, 1),'x77': (0, 1),'x78': (0, 1),'x79': (0, 1),'x80': (0, 1),\
        # 'x81': (0, 1),'x82': (0, 1),'x83': (0, 1),'x84': (0, 1),'x85': (0, 1),\
        # 'x86': (0, 1),'x87': (0, 1),'x88': (0, 1),'x89': (0, 1),'x90': (0, 1),\
        # 'x91': (0, 1),'x92': (0, 1),'x93': (0, 1),'x94': (0, 1),'x95': (0, 1),\
        # 'x96': (0, 1),'x97': (0, 1),'x98': (0, 1),'x99': (0, 1),'x100': (0, 1)}
        optimizer = BayesianOptimization(
            # f=data_hyper_clean_BO.black_box_function,
            f=data_hyper_clean_BO.black_box_function,
            pbounds=pbounds,
            random_state=1,
        )

        partition = 20
        part_lmd=np.ones(partition, dtype='float32')*0.1
        optimizer.probe(
            params=part_lmd,
            lazy=True,
        )

        # Will probe only the two points specified above
        optimizer.maximize(init_points=0, n_iter=0)
        # Bounded region of parameter space
        
        
        optimizer.maximize(
            init_points=10,
            n_iter=500,
            alpha=1e-3,
            acq="ucb", kappa=10
        )
        print('BO time', time.time()-bo_start)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))


if __name__ == '__main__':
    # np.random.seed(0)
    # data = load_std_data()
    # print(data.train.info['noisy examples'])
    # print(data.train.info['clean examples'])
    main(run_baseline=False, run_oracle=False, run_optimization=True,
         run_bayesian=True,T=200,lr=0.1)