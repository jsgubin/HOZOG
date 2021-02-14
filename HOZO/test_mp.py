#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from rfho.datasets import load_mnist
import hozo
import time

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

def main(saver=None, run_optimization=True, T=2000, lr=.1):
    """
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Ban gpus
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    num_gpus = 4

    np.random.seed(0)
    data = load_std_data(sets_proportions=(.5, .45),limit_ex=4000)    

    if run_optimization:
        partition = data.train.num_examples
        times = int(data.train.num_examples/partition)
        lambda0 = np.ones(partition, dtype='float32')*0.1
        data_hyper_clean = hozo.HOZO(model=hozo.data_cleaning_mode, 
            max_iter=2000, 
            eta=40,
            q=5, 
            mu=1e-3)

        init_model_dict = {'num_gpus':num_gpus,'T':T,'lr':lr, 'times':times}

        data_dict = {'data':data}

        kw = {'init_model_dict':init_model_dict, 'data_dict':data_dict}

        data_hyper_clean.fit(lmd0=lambda0, 
            name='test_mp',
            **kw)


if __name__ == '__main__':
    main(run_optimization=True, T=2000, lr=0.1)