import numpy as np
from scipy import sparse
from sklearn import linear_model
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import time
import hozo
import cvxopt
from cvxopt import spmatrix, matrix, sparse


class ZerothOrderGradient():

    def __init__(self, 
        black_box_function,
        lambda0=None, 
        tol=0.1, 
        callback=None,
        max_iter=100,
        mu =0.01, 
        q=5, 
        eta=0.001, 
        rv_dist='UnitBall'):
        self.loss = 1e10
        if lambda0 is None: 
            raise ValueError('lambda0 must be init!')
        self.lambdak = lambda0
        self.tol = tol
        self.callback = callback
        self.max_iter = max_iter
        self.mu = mu
        self.q = q
        self.eta = eta
        self.times = []
        self.losses = []
        self.lambdas = []
        self.grads = []

        if(rv_dist == 'UnitBall'):
            self.RV_Gen = self.Draw_UnitBall
        elif(rv_dist == 'UnitSphere'):
            self.RV_Gen = self.Draw_UnitSphere
        else:
            print('Please specify a valid distribution for random perturbation')

        self.black_box_function = black_box_function

    def fit(self):
        time_start = time.time()

        best_loss = 1e10;
        best_lambda = self.lambdak
        lambdak = self.lambdak
        print('start')
        old_loss = self.black_box_function(lambdak)
        print('old_loss',old_loss)
        for iter in range(self.max_iter):
            self.losses.append(old_loss)
            self.lambdas.append(lambdak)
            zo_gradient = self.full_zo_gradient_estimation_oldloss(lmd=lambdak,old_loss=old_loss)
            self.grads.append(zo_gradient)
            lambdak = lambdak - self.eta * zo_gradient
            old_loss = self.black_box_function(lambdak)
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('losses',self.losses)

            if(old_loss < best_loss):
                best_loss = old_loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
        self.lambdak = best_lambda
        return self

    def Draw_UnitBall(self, size):
        sample = np.random.uniform(-1.0, 1.0, size=size)
        return sample/np.linalg.norm(sample.flatten())

    def Draw_UnitSphere(self, size):
        sample = np.random.normal(0.0, 1.0, size=size)
        return sample/np.linalg.norm(sample.flatten())

    def full_zo_gradient_estimation_oldloss(self, lmd, old_loss):
        mu = self.mu
        q = self.q
        f = old_loss
        grad_avg = np.zeros(lmd.size)
        for q_idx in range(q):
            u_rand = self.RV_Gen(lmd.size)
            tmp_lambdak = lmd + mu * u_rand
            # pj_lambdak = projection(tmp_lambdak)
            # u_rand = (pj_lambdak - lmd)/mu
            # print('tmp_lambdak',tmp_lambdak)
            f_perturb = self.black_box_function(tmp_lambdak)
            grad_avg += (f_perturb-f)*u_rand
        return (lmd.size/mu)*(grad_avg/q)