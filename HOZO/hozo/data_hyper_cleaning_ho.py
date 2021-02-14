import numpy as np
from scipy import sparse
from sklearn import linear_model
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import hozo
import rfho as rf
import cvxopt
from cvxopt import spmatrix, matrix, sparse


class DataHyperCleaning(linear_model.base.BaseEstimator,
                           linear_model.base.LinearClassifierMixin):

    def __init__(
                 self, lambda0=None, tol=0.1, inner_tol=1e-4, callback=None,
                 max_iter=100, mu =0.01, q=1, eta=0.001, rv_dist='UnitSphere', loss_function='ce'):
        self.loss = 1e10
        if lambda0 is None: 
            raise ValueError('lambda0 must be init!')
        self.lambdak = lambda0
        self.tol = tol
        self.inner_tol = inner_tol
        self.callback = callback
        self.max_iter = max_iter
        self.mu = mu
        self.q = q
        self.eta = eta
        self.times = []
        self.losses = []
        self.lambdas = []

        if(rv_dist == 'UnitBall'):
            self.RV_Gen = self.Draw_UnitBall
        elif(rv_dist == 'UnitSphere'):
            self.RV_Gen = self.Draw_UnitSphere
        else:
            print('Please specify a valid distribution for random perturbation')

        if(loss_function == 'ce'):
            self.cost_func = self.cost_func_ce_old
        elif(loss_function == '01'):
            self.cost_func = self.cost_func_01
        else:
            print('Please specify a valid loss function')

    def fit(self, data, saver=None, T=2000, lr=.1, projection=None):
        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        best_loss = 1e10;
        best_lambda = self.lambdak
        lambdak = self.lambdak
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            loss, zo_gradient = self.full_zo_gradient_estimation(saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
            # print('Current lambdak:',lambdak)
            print('zo_gradient',zo_gradient)
            lambdak = lambdak - self.eta * zo_gradient
            print('lambdak',lambdak)
            lambdak = projection(lambdak)
            # if lambdak < 0:
            #     lambdak = [1e-20]
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',loss)
            if(loss < best_loss):
                best_loss = loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            self.losses.append(loss)
            print('losses',self.losses)

        self.lambdak = best_lambda
        return self


    def fit_partition(self, data, part_lmd, saver=None, T=2000, lr=.1, projection=None,name='test'):
        projection = self._get_projector(R=part_lmd.size, N_ex=part_lmd.size)

        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        gaps = []
        zog = []
        tr_loss = []
        t_loss = []
        eta = self.eta

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        times = int(lambdak.size/part_lmd.size)
        print('start')

        old_loss,old_train,old_test = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')

        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            self.losses.append(old_loss)
            tr_loss.append(old_train)
            t_loss.append(old_test)
            print('losses',self.losses)
            print('tr_loss',tr_loss)
            print('t_loss',t_loss)

            avg_gap, zo_gradient, part_zo_gradient = \
        self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
            # zog_norm=np.linalg.norm(zo_gradient)
            # print('data type:',zo_gradient.dtype)
            # print('data type:',part_zo_gradient.dtype)
            part_lmd2 = part_lmd - self.eta *  part_zo_gradient
            # lambdak2 = lambdak - self.eta * zo_gradient
            # print('data type:',part_lmd2.dtype)
            # print('data type:',part_lmd2.dtype)
            # part_lmd2 = projection(part_lmd2)
            lambdak2 = np.tile(part_lmd2, times)
            # lambdak2 = projection(lambdak2)
            if (part_lmd2==lambdak2).all:
                print('equal')

            new_loss,new_train,new_test = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
            
            print('new_loss',new_loss)
            zog.append(zo_gradient)
            np.savetxt("./experiments/zog_eta%f_q%d_mu%f%s.txt"%(self.eta,self.q,self.mu,name),zog)
            np.savetxt("./experiments/lmd_eta%f_q%d_mu%f%s.txt"%(self.eta,self.q,self.mu,name),self.lambdas)

            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            old_train = new_train
            old_test = new_test

            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',old_loss)

            if(old_loss < best_loss):
                best_loss = old_loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            gaps.append(avg_gap)
            print('times',self.times)
            print('gaps',gaps)


        self.lambdak = best_lambda
        return self

    def fit_partition_only(self, data, part_lmd, saver=None, T=2000, lr=.1, projection=None,name='test'):
        projection = self._get_projector(R=part_lmd.size, N_ex=part_lmd.size)

        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        times = int(lambdak.size/part_lmd.size)
        print('start')

        old_loss,old_train,old_test = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')

        for iter in range(self.max_iter):

            avg_gap, zo_gradient, part_zo_gradient = \
        self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
            part_lmd2 = part_lmd - self.eta *  part_zo_gradient
            tmp_time=time.time()
            part_lmd2 = projection(part_lmd2)
            print('Projection time',time.time()-tmp_time)
            lambdak2 = np.tile(part_lmd2, times)

            tmp_time=time.time()
            new_loss,new_train,new_test = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
            print('Evaluation time',time.time()-tmp_time)

            print('new_loss',new_loss)

            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            old_train = new_train
            old_test = new_test

            if(iter%10 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',old_loss)

            if(old_loss < best_loss):
                best_loss = old_loss
                best_lambda = lambdak
            # print('times',time.time()-time_start)

        self.lambdak = best_lambda
        return self

    def fit_partition_ADAM(self, 
        data, 
        part_lmd, 
        saver=None, 
        T=2000, 
        lr=.1, 
        projection=None,
        beta1=.9, 
        beta2=.999, 
        eps=1.e-6,
        name='test'):
        projection = self._get_projector(R=part_lmd.size, N_ex=part_lmd.size)

        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        gaps = []
        zog = []
        tr_loss = []
        t_loss = []
        eta = self.eta

        # Init for Adam
        m = np.zeros(part_lmd.size, dtype='float32')
        v = np.zeros(part_lmd.size, dtype='float32')

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        times = int(lambdak.size/part_lmd.size)
        print('start')

        old_loss,old_train,old_test = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')

        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            self.losses.append(old_loss)
            tr_loss.append(old_train)
            t_loss.append(old_test)
            print('losses',self.losses)
            print('tr_loss',tr_loss)
            print('t_loss',t_loss)

            avg_gap, zo_gradient, part_zo_gradient = \
        self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)

            m = beta1 * m + (1. - beta1) * part_zo_gradient
            v = beta2 * v + (1. - beta2) * np.square(part_zo_gradient)

            bias_correction = np.sqrt(1. - np.power(beta2, float(iter+1))) / (
            1. - np.power(beta1, float(iter+1)))
            eta = eta * bias_correction

            print('eta', eta)

            v_epsilon_k = v + eps
            v_tilde_k = np.sqrt(v_epsilon_k)  # + eps

            part_lmd2 = part_lmd - eta * (beta1 * m + (1. - beta1) * part_zo_gradient) / v_tilde_k
            # lambdak2 = lambdak - eta * (beta1 * m + (1. - beta1) * zo_gradient) / v_tilde_k

            # part_lmd2 = projection(part_lmd2)
            # lambdak2 = projection(lambdak2)
            lambdak2 = np.tile(part_lmd2, times)

            # print(part_lmd2.dtype)
            # print(lambdak2.dtype)
            new_loss,new_train,new_test = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
            
            print('new_loss',new_loss)
            zog.append(zo_gradient)
            np.savetxt("./experiments/zog_eta%f_q%d_mu%f.txt"%(self.eta,self.q,self.mu),zog)
            np.savetxt("./experiments/lmd_eta%f_q%d_mu%f.txt"%(self.eta,self.q,self.mu),self.lambdas)

            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            old_train = new_train
            old_test = new_test

            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',old_loss)

            if((iter+1)%3 == 0):
                m = np.zeros(part_lmd.size, dtype='float32')
                v = np.zeros(part_lmd.size, dtype='float32')
                # eta = self.eta * (0.9 **iter)
                # eta = self.eta
                print('reset')

            if(old_loss < best_loss):
                best_loss = old_loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            gaps.append(avg_gap)
            print('times',self.times)
            print('gaps',gaps)


        self.lambdak = best_lambda
        return self

    def fit_partition_ADAM_log(self, 
        data, 
        part_lmd, 
        saver=None, 
        T=2000, 
        lr=.1, 
        projection=None,
        beta1=.9, 
        beta2=.999, 
        eps=1.e-6):
        projection = self._get_projector(R=part_lmd.size, N_ex=part_lmd.size)

        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        gaps = []
        zog = []
        eta = self.eta

        # Init for Adam
        m = np.zeros(part_lmd.size)
        v = np.zeros(part_lmd.size)

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        print('start')
        
        old_loss,old_train,old_test = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')
        # old_loss,old_train,old_w,old_b = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')
        # old_loss2 = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')
        print('old_loss',old_loss)
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            # print(lambdak)
            
            # print('Current lambdak:',lambdak)
            # print('zo_gradient',zo_gradient)
            
            # new_loss = 1e10
            # while new_loss>old_loss+0.002:
            print('gaps',gaps)
            avg_gap, avg_gap2, avg_gap3,avg_gap4, part_u_rand, zo_gradient, part_zo_gradient = \
        self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss,old_train=old_train, old_w=old_w,old_b=old_b, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
            # print('part_zo_gradient',part_zo_gradient)
            # self.mu=self.mu*2
            # print(self.mu)
            # if avg_gap>0.1:
            #     np.savetxt("./experiments/lmd",part_lmd)
            #     np.savetxt("./experiments/urand",part_u_rand)
            # # gaps.append(avg_gap)
            # gaps.append(avg_gap2)
            # gaps.append(avg_gap3)
            # gaps.append(avg_gap4)
        #     avg_gap, avg_gap2, part_u_rand, zo_gradient, part_zo_gradient = \
        # self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss,old_train=old_train, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
        #     gaps.append(avg_gap)
        #     gaps.append(avg_gap2)
        #     self.mu=self.mu/2
        #     print(self.mu)
            # print(np.mean(gaps))
            m = beta1 * m + (1. - beta1) * part_zo_gradient
            v = beta2 * v + (1. - beta2) * np.square(part_zo_gradient)

            bias_correction = np.sqrt(1. - np.power(beta2, float(iter+1))) / (
            1. - np.power(beta1, float(iter+1)))
            self.eta = self.eta * bias_correction

            v_epsilon_k = v + eps
            v_tilde_k = np.sqrt(v_epsilon_k)  # + eps

            part_lmd2 = part_lmd - self.eta * (beta1 * m + (1. - beta1) * part_zo_gradient) / v_tilde_k
            lambdak2 = lambdak - self.eta * (beta1 * m + (1. - beta1) * zo_gradient) / v_tilde_k

            # part_u_rand = self.RV_Gen(part_lmd.size)
            # part_lmd2 = part_lmd2+part_u_rand
            # lambdak2 = lambdak2+part_u_rand

            # print('lambdak2',lambdak2)
            # print('part_lmd2',part_lmd2)
            part_lmd2 = projection(part_lmd2)
            lambdak2 = projection(lambdak2)
            if (part_lmd2==part_lmd2).all:
                print('equal')

            # print(lambdak2)
            new_loss,new_train,new_w,new_b = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
            # new_loss2 = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
            print('new_loss',new_loss)
                # if new_loss>old_loss+0.002:
                #     self.eta = self.eta*0.5
                #     print('Eta',self.eta)
                # else:
                #     self.eta = self.eta*1.01
                #     print('Eta',self.eta)

            zog.append(zo_gradient)

            np.savetxt("./experiments/zog_eta%f_q%d_mu%f.txt"%(eta,self.q,self.mu),zog)
            np.savetxt("./experiments/lmd_eta%f_q%d_mu%f.txt"%(eta,self.q,self.mu),self.lambdas)

            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            old_train = new_train
            old_w = new_w
            old_b = new_b
            # if lambdak < 0:
            #     lambdak = [1e-20]
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',old_loss)
            # if(loss < old_loss):
            #     self.eta = self.eta*1.01
            # else:
            #     self.eta = self.eta*0.5
            # old_loss = loss
            # print(self.eta)
            if(old_loss < best_loss):
                best_loss = old_loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            self.losses.append(old_loss)
            print('losses',self.losses)
            print('times',self.times)


        self.lambdak = best_lambda
        return self

    def fit_partition_NAG(self, data, part_lmd, saver=None, T=2000, lr=.1, discount=.7, projection=None):
        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        print('start')
        part_pre_grad = np.zeros(part_lmd.size)
        pre_grad = np.zeros(lambdak.size)

        old_loss = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            # print(lambdak)
            
            # print('Current lambdak:',lambdak)
            # print('zo_gradient',zo_gradient)
            
            new_loss = 1e10
            
            while new_loss>old_loss+0.002:
                part_pre_grad2 = part_pre_grad
                pre_grad2 = pre_grad
                # print('part_pre_grad2',part_pre_grad2)
                # print('pre_grad2',pre_grad2)

                part_lambda_future = part_lmd - self.eta*discount*part_pre_grad2
                lambda_future = lambdak - self.eta*discount*pre_grad2

                part_lambda_future = projection(part_lambda_future)
                lambda_future = projection(lambda_future)

                loss, zo_gradient, part_zo_gradient = \
            self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss, part_lmd=part_lambda_future, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambda_future, projection=projection)
                # print('part_zo_gradient',part_zo_gradient)
                # print('zo_gradient',zo_gradient)
                
                pre_grad2 = pre_grad2*discount+zo_gradient
                part_pre_grad2 = part_pre_grad2*discount+part_zo_gradient

                part_lmd2 = part_lmd - self.eta * part_pre_grad2
                lambdak2 = lambdak - self.eta * pre_grad2
                # print('lambdak2',lambdak2)
                print('part_lmd2',part_lmd2)
                print('lambdak2',lambdak2)
                part_lmd2 = projection(part_lmd2)
                lambdak2 = projection(lambdak2)
                new_loss = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
                print('new_loss',new_loss)
                # if new_loss>old_loss+0.002:
                #     self.eta = self.eta*0.5
                #     print('Eta',self.eta)
                # else:
                #     self.eta = self.eta*1.01
                #     print('Eta',self.eta)
            
            part_pre_grad = part_pre_grad2
            pre_grad = pre_grad2
            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            # if lambdak < 0:
            #     lambdak = [1e-20]
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',loss)
            # if(loss < old_loss):
            #     self.eta = self.eta*1.01
            # else:
            #     self.eta = self.eta*0.5
            # old_loss = loss
            # print(self.eta)
            if(loss < best_loss):
                best_loss = loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            self.losses.append(loss)
            print('losses',self.losses)
            print('times',self.times)


        self.lambdak = best_lambda
        return self

    def fit_partition_coordinate(self, data, part_lmd, saver=None, T=2000, lr=.1, cd_size = 50, projection=None):
        # projection = self._get_projector(R=part_lmd.size, N_ex=part_lmd.size)

        time_start = time.time()

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = hozo.LinearModel(x, 28 * 28, 10)

        best_loss = 1e10
        best_lambda = self.lambdak
        lambdak = self.lambdak
        print('start')
        old_loss = self.cost_func(saver, model, y, data, T, lr, lambdak, name='Old loss')
        print('old_loss',old_loss)
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            # print(lambdak)
            
            # print('Current lambdak:',lambdak)
            # print('zo_gradient',zo_gradient)
            
            new_loss = 1e10
            while new_loss>old_loss+0.002:
                loss, zo_gradient, part_zo_gradient = \
            self.part_full_zo_gradient_estimation_oldloss_cd(old_loss=old_loss, cd_size=cd_size, part_lmd=part_lmd, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambdak, projection=projection)
                # print('part_zo_gradient',part_zo_gradient)
                part_lmd2 = part_lmd - self.eta * part_zo_gradient
                lambdak2 = lambdak - self.eta * zo_gradient
                # print('lambdak2',lambdak2)
                # print('part_lmd2',part_lmd2)
                part_lmd2 = projection(part_lmd2)
                lambdak2 = projection(lambdak2)
                print('part_lmd2',part_lmd2)
                new_loss = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
                print('new_loss',new_loss)
                if new_loss>old_loss+0.002:
                    self.eta = self.eta*0.5
                    print('Eta',self.eta)
                else:
                    self.eta = self.eta*1.01
                    print('Eta',self.eta)
            
            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            # if lambdak < 0:
            #     lambdak = [1e-20]
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',loss)
            # if(loss < old_loss):
            #     self.eta = self.eta*1.01
            # else:
            #     self.eta = self.eta*0.5
            # old_loss = loss
            # print(self.eta)
            if(loss < best_loss):
                best_loss = loss
                best_lambda = lambdak
            self.times.append(time.time()-time_start)
            self.losses.append(loss)
            print('losses',self.losses)
            print('times',self.times)


        self.lambdak = best_lambda
        return self

    def Draw_UnitBall(self, size):
        sample = np.random.uniform(-1.0, 1.0, size=size).astype(np.float32)
        return sample/np.linalg.norm(sample.flatten())

    def Draw_UnitSphere(self, size):
        sample = np.random.normal(0.0, 1.0, size=size).astype(np.float32)
        return sample/np.linalg.norm(sample.flatten())

    def full_zo_gradient_estimation(self, saver, model, y, data, T, lr, lmd, projection, name='zo gradient'):
        mu = self.mu
        q = self.q
        f = self.cost_func(saver, model, y, data, T, lr, lmd, name)
        grad_avg = np.zeros(lmd.size)
        for q_idx in range(q):
            u_rand = self.RV_Gen(lmd.size)
            tmp_lambdak = lmd + mu * u_rand
            # pj_lambdak = projection(tmp_lambdak)
            # u_rand = (pj_lambdak - lmd)/mu
            # print('tmp_lambdak',tmp_lambdak)
            f_perturb = self.cost_func(saver, model, y, data, T, lr, tmp_lambdak, name)
            grad_avg += (f_perturb-f)*u_rand
        return f,(lmd.size/mu)*(grad_avg/q)

    def part_full_zo_gradient_estimation(self, part_lmd, saver, model, y, data, T, lr, lmd, projection, name='zo gradient'):
        mu = self.mu
        q = self.q
        # print(part_lmd)
        # print(lmd)
        f = self.cost_func(saver, model, y, data, T, lr, lmd, name)
        grad_avg = np.zeros(lmd.size)
        part_grad_avg = np.zeros(part_lmd.size)
        print(part_lmd)
        for q_idx in range(q):
            part_u_rand = self.RV_Gen(part_lmd.size)
            tmp_part_lambdak = part_lmd + mu * part_u_rand
            part_size = int(np.ceil(lmd.size/part_lmd.size))
            # print(part_size)
            tmp_lambdak=np.ones(lmd.size)
            for p_idx in range(part_lmd.size):
                if p_idx == part_lmd.size-1:

                    tmp_lambdak[part_size*p_idx:]=np.ones(lmd.size-(part_lmd.size-1)*part_size)*tmp_part_lambdak[p_idx]
                else:
                    # print(part_size*p_idx)
                    # print(part_size*(p_idx+1)-1)
                    # print(tmp_part_lambdak[p_idx])
                    tmp_lambdak[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size)*tmp_part_lambdak[p_idx]    

            # pj_lambdak = projection(tmp_lambdak)
            # u_rand = (pj_lambdak - lmd)/mu
            # print('tmp_lambdak',tmp_lambdak)
            u_rand = (tmp_lambdak - lmd)/mu
            # print('u_rand',u_rand)
            # print('part_u_rand',part_u_rand)
            f_perturb = self.cost_func(saver, model, y, data, T, lr, tmp_lambdak, name)
            grad_avg += (f_perturb-f)*u_rand
            part_grad_avg += (f_perturb-f)*part_u_rand
        return f, (part_lmd.size/mu)*(grad_avg/q), (part_lmd.size/mu)*(part_grad_avg/q)

    def part_full_zo_gradient_estimation_oldloss(self, old_loss, part_lmd, saver, model, y, data, T, lr, lmd, projection, name='zo gradient'):
        mu = self.mu
        q = self.q
        f = old_loss
        avg_gap = 0.
        times = int(lmd.size/part_lmd.size)
        # print('data type avg_gap:',type(avg_gap))
        grad_avg = np.zeros(lmd.size,dtype='float32')
        part_grad_avg = np.zeros(part_lmd.size,dtype='float32')
        # print('part_lmd',part_lmd)
        # print('data type plmd:',part_lmd.dtype)
        # print('data type lmd:',lmd.dtype)
        for q_idx in range(q):
            part_u_rand = self.RV_Gen(part_lmd.size)
            # part_u_rand = np.zeros(part_lmd.size,dtype='float32')
            # print('data type:',part_u_rand.dtype)
            # tmp_part_lambdak = part_lmd + mu * part_u_rand
            # print('data type2:',tmp_part_lambdak.dtype)
            # tmp_lambdak = np.tile(tmp_part_lambdak, times)
            # print('part_u_rand',part_u_rand)
            u_rand = np.tile(part_u_rand, times)
            tmp_lambdak = lmd + mu * u_rand
            # tmp_lambdak = lmd
            # print('data type3:',u_rand.dtype)

            f_perturb,_,_ = self.cost_func(saver, model, y, data, T, lr, tmp_lambdak, name)
            grad_avg += (f_perturb-f)*u_rand
            avg_gap += (f_perturb-f)
            # print('valid_gap',f_perturb-f)
            part_grad_avg += (f_perturb-f)*part_u_rand
            # print('data type f_perturb:',type(f_perturb))

        return avg_gap/q, (part_lmd.size/mu)*(grad_avg/q), (part_lmd.size/mu)*(part_grad_avg/q)

    def part_full_zo_gradient_estimation_oldloss_log(self, old_loss,old_train,old_w,old_b, part_lmd, saver, model, y, data, T, lr, lmd, projection, name='zo gradient'):
        mu = self.mu
        q = self.q
        # print(part_lmd)
        # print(lmd)
        f = old_loss
        f2 = old_train
        avg_gap = 0.
        avg_gap2 = 0.
        avg_gap3 = 0.
        avg_gap4 = 0.

        grad_avg = np.zeros(lmd.size)
        part_grad_avg = np.zeros(part_lmd.size)
        print('part_lmd',part_lmd)
        for q_idx in range(q):
            part_u_rand = self.RV_Gen(part_lmd.size)
            tmp_part_lambdak = part_lmd + mu * part_u_rand
            print(mu * part_u_rand)
            part_size = int(np.ceil(lmd.size/part_lmd.size))
            # print(part_size)
            tmp_lambdak=np.ones(lmd.size)
            for p_idx in range(part_lmd.size):
                if p_idx == part_lmd.size-1:

                    tmp_lambdak[part_size*p_idx:]=np.ones(lmd.size-(part_lmd.size-1)*part_size)*tmp_part_lambdak[p_idx]
                else:
                    # print(part_size*p_idx)
                    # print(part_size*(p_idx+1)-1)
                    # print(tmp_part_lambdak[p_idx])
                    tmp_lambdak[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size)*tmp_part_lambdak[p_idx]    

            # pj_lambdak = projection(tmp_lambdak)
            # u_rand = (pj_lambdak - lmd)/mu
            print('tmp_lambdak',tmp_lambdak)
            # u_rand = (tmp_lambdak - lmd)/mu
            u_rand = self.extend_vector(part_u_rand, lmd.size)
            if (part_u_rand==u_rand).all():
                print('-------------------------------------')
            # print('u_rand',u_rand)
            # print('part_u_rand',part_u_rand)
            # f_perturb,f_perturb2 = self.cost_func(saver, model, y, data, T, lr, tmp_lambdak, name)
            f_perturb,f_perturb2,w3,b3 = self.cost_func(saver, model, y, data, T, lr, tmp_part_lambdak, name)
            grad_avg += (f_perturb-f)*u_rand
            avg_gap += (f_perturb-f)
            avg_gap2 += (f_perturb2-f2)
            avg_gap3 += np.linalg.norm(w3-old_w)
            avg_gap4 += np.linalg.norm(b3-old_b)
            if f_perturb-f>0:
                print('f, ff %e, %e'%(f,f_perturb))
            print('gap1',f_perturb-f)
            print('gap2',f_perturb2-f2)
            print('gap3 %e'%np.linalg.norm(w3-old_w))
            print('gap4 %e'%np.linalg.norm(b3-old_b))
            if not ((w3==old_w).all()):
                np.savetxt("./experiments/w.txt",w3)
                np.savetxt("./experiments/w.txt",old_w)
            part_grad_avg += (f_perturb-f)*part_u_rand
        return avg_gap/q,avg_gap2/q,avg_gap3/q,avg_gap4/q, part_u_rand, (part_lmd.size/mu)*(grad_avg/q), (part_lmd.size/mu)*(part_grad_avg/q)

    def part_full_zo_gradient_estimation_oldloss_cd(self, old_loss, cd_size, part_lmd, saver, model, y, data, T, lr, lmd, projection, name='zo gradient'):
        mu = self.mu
        q = self.q
        # print(part_lmd)
        # print(lmd)
        f = old_loss
        grad_avg = np.zeros(lmd.size)
        part_grad_avg = np.zeros(part_lmd.size)
        # print('part_lmd',part_lmd)
        random_indices = np.sort(np.random.permutation(part_lmd.size)[
                            :int(cd_size*q)])
        for q_idx in range(q):

            tmp_indices = random_indices[int(q_idx*cd_size):int((q_idx+1)*cd_size)]

            # print(tmp_indices.size)

            part_u_rand_cd = self.RV_Gen(int(cd_size))

            part_u_rand = np.zeros(part_lmd.size)

            for ui in range(cd_size):
                ti = tmp_indices[ui]
                part_u_rand[ti] = part_u_rand[ti] + part_u_rand_cd[ui]

            tmp_part_lambdak = part_lmd + mu * part_u_rand
            part_size = int(np.ceil(lmd.size/part_lmd.size))
            # print(part_size)
            tmp_lambdak=np.ones(lmd.size)
            for p_idx in range(part_lmd.size):
                if p_idx == part_lmd.size-1:

                    tmp_lambdak[part_size*p_idx:]=np.ones(lmd.size-(part_lmd.size-1)*part_size)*tmp_part_lambdak[p_idx]
                else:
                    # print(part_size*p_idx)
                    # print(part_size*(p_idx+1)-1)
                    # print(tmp_part_lambdak[p_idx])
                    tmp_lambdak[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size)*tmp_part_lambdak[p_idx]    

            # pj_lambdak = projection(tmp_lambdak)
            # u_rand = (pj_lambdak - lmd)/mu
            # print('tmp_lambdak',tmp_lambdak)
            u_rand = (tmp_lambdak - lmd)/mu
            # print('u_rand',u_rand)
            # print('part_u_rand',part_u_rand)
            f_perturb = self.cost_func(saver, model, y, data, T, lr, tmp_lambdak, name)
            grad_avg += (f_perturb-f)*u_rand
            part_grad_avg += (f_perturb-f)*part_u_rand
        return f, (part_lmd.size/mu)*(grad_avg/q), (part_lmd.size/mu)*(part_grad_avg/q)

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
        tf.set_random_seed(1)
        def g_logits(x,y):
            # init_random = tf.contrib.layers.xavier_initializer(seed=1)
            # h1 = layers.fully_connected(x, 300, weights_initializer=init_random,biases_initializer=init_random)
            # logits = layers.fully_connected(h1, int(y.shape[1]), weights_initializer=init_random,biases_initializer=init_random)
            # with tf.variable_scope('model'):
            h1 = layers.fully_connected(x, 300)
            logits = layers.fully_connected(h1, int(y.shape[1]))
            return logits
        x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
        y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
        logits = g_logits(x,y)

        train_s = data.train.create_supplier(x, y)
        test_s = data.test.create_supplier(x, y)
        valid_s = data.validation.create_supplier(x, y)
        # print(data.train.target)

        lmd = tf.convert_to_tensor(lmd, dtype=tf.float32)
        # lmd2 = tf.Variable(lmd,
        #               dtype=tf.float32)
        # lmd2.assign(lmd)

        
        # ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        error2 = tf.reduce_mean(tf.sigmoid(lmd)*tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

        opt = tf.train.GradientDescentOptimizer(lr)
        ts1 = opt.minimize(error2)

        with tf.Session(config=hozo.CONFIG_GPU_GROWTH).as_default() as sess:
            tf.global_variables_initializer().run()
            # lmd1=model.Ws[0].eval()
            for t_idx in range(T):
                ts1.run(feed_dict=train_s())
                # train_error = error2.eval(feed_dict=train_s())
                # test_error = error.eval(feed_dict=test_s())
                # valid_error = error.eval(feed_dict=valid_s())
                # print(train_error)
            # print(train_error)
                # lmd2=model.Ws[0].eval()
                # tmp_grads = (lmd1 - lmd2) / lr
                # lmd1=lmd2
                # grads_norm=np.linalg.norm(tmp_grads)
                # if grads_norm<0.01:
                #     pass
                #     # print(t_idx)
                #     # break

            # if saver: saver.save(name)
            # baseline_test_accuracy = accuracy2.eval(feed_dict=valid_s())
            train_error = error2.eval(feed_dict=train_s())
            test_error = error.eval(feed_dict=test_s())
            valid_error = error.eval(feed_dict=valid_s())
            # print(train_error)
            # print('baseline_test_accuracy',baseline_test_accuracy)
            # return 1-baseline_test_accuracy
            # return test_error, train_error, test_s, model.Ws[0].eval(),model.bs[0].eval()
            print(train_error)
            print(tf.sigmoid(lmd).eval())
            return valid_error, train_error, test_error

    def cost_func_ce_old2(self, saver, model, y, data, T, lr, lmd=None, name=None):
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
        x = model.inp[0]

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            # tf.set_random_seed(0)
            x = tf.placeholder(tf.float32, name='x')
            y = tf.placeholder(tf.float32, name='y')

            # W = tf.Variable(tf.zeros([784, 10]), name='W')
            # b = tf.Variable(tf.zeros([10]), name='b')

            # out = tf.identity(tf.matmul(x, W) + b)

            model = hozo.LinearModel(x, 28 * 28, 10)


            train_s = data.train.create_supplier(x, y)
            test_s = data.test.create_supplier(x, y)
            valid_s = data.validation.create_supplier(x, y)

            # error2 = tf.reduce_mean(lmd * hozo.cross_entropy_loss(y, model.out))
            ce = hozo.cross_entropy_loss(y, model.out)
            # ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model.out)
            error2 = tf.reduce_mean(tf.sigmoid(lmd) * ce)
            correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
            accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
            error = tf.reduce_mean(ce)

            opt = tf.train.GradientDescentOptimizer(lr)
            ts1 = opt.minimize(error2, var_list=model.var_list)
            # ts1 = opt.minimize(error2, var_list=[W,b])

            # print(tf.get_default_graph().as_graph_def())
            with tf.Session(config=hozo.CONFIG_GPU_GROWTH).as_default() as sess:
                # tf.variables_initializer([W,b]).run()
                tf.variables_initializer(model.var_list).run()
                # tf.global_variables_initializer().run()
                # print(W.eval())
                # lmd1=model.Ws[0].eval()
                for t_idx in range(T):
                    ts1.run(feed_dict=train_s())

                    # lmd2=model.Ws[0].eval()
                    # tmp_grads = (lmd1 - lmd2) / lr
                    # lmd1=lmd2
                    # train_error = error2.eval(feed_dict=train_s())
                    # print(train_error)
                    # tmp_grads2=tf.gradients(error2,model.Ws[0])[0].eval(feed_dict=train_s())
                    
                    # grads_and_vars = opt.compute_gradients(error2, var_list=model.var_list)
                    # grad_SSEs = [tf.reduce_sum(var**2) for var in grads_and_vars]
                    # print(sess.run(grads_and_vars,feed_dict=train_s()))
                    # opt.apply_gradients(grads_and_vars).run(feed_dict=train_s())
                    # update=opt.apply_gradients(grads_and_vars)
                    # sess.run((grads_and_vars,update),feed_dict=train_s())
                    # grad_t = sess.run(grads_and_vars, feed_dict=train_s())
                    # sess.run(update, feed_dict=train_s())
                    # print(grads_and_vars)
                    # tmp_grads = grads_and_vars[0][0].eval()
                    # print(tmp_grads)
                    # grads_norm=np.linalg.norm(tmp_grads)
                    # print(grads_norm)
                    # if grads_norm<0.01:
                    #     pass
                        # print(t_idx)
                        # break

                # baseline_test_accuracy = accuracy2.eval(feed_dict=valid_s())
                # print(tf.get_default_graph().as_graph_def())
                train_error = error2.eval(feed_dict=train_s())
                test_error = error.eval(feed_dict=test_s())
                valid_error = error.eval(feed_dict=valid_s())
                # print(train_error)
                # writer = tf.summary.FileWriter('./graphs2', graph)
                # print('baseline_test_accuracy',baseline_test_accuracy)
                # return 1-baseline_test_accuracy
                # return test_error, train_error, test_s, model.Ws[0].eval(),model.bs[0].eval()
                # print(valid_error)
                return valid_error, train_error, test_error

    def cost_func_ce_old(self, saver, model, y, data, T, lr, lmd=None, name=None):
        x = model.inp[0]

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            x = tf.placeholder(tf.float32, name='x')
            y = tf.placeholder(tf.float32, name='y')

            W = tf.Variable(tf.zeros([784, 10],tf.float32), name='W')
            b = tf.Variable(tf.zeros([10],tf.float32), name='b')

            out = tf.matmul(x, W) + b

            train_s = data.train.create_supplier(x, y)
            test_s = data.test.create_supplier(x, y)
            valid_s = data.validation.create_supplier(x, y)

            ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out)
            error2 = tf.reduce_mean(tf.sigmoid(lmd) * ce)
            correct_prediction2 = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
            accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
            error = tf.reduce_mean(ce)

            opt = tf.train.GradientDescentOptimizer(lr)
            ts1 = opt.minimize(error2, var_list=[W,b])
            ts1 = opt.minimize(error2)

            # print(tf.get_default_graph().as_graph_def())
            with tf.Session(config=hozo.CONFIG_GPU_GROWTH).as_default() as sess:
                # tf.variables_initializer([W,b]).run()
                tf.global_variables_initializer().run()
                # print(W.eval())
                # lmd1=model.Ws[0].eval()
                for t_idx in range(T):
                    ts1.run(feed_dict=train_s())

                    # lmd2=model.Ws[0].eval()
                    # tmp_grads = (lmd1 - lmd2) / lr
                    # lmd1=lmd2
                    # train_error = error2.eval(feed_dict=train_s())
                    # print(train_error)
                    # tmp_grads2=tf.gradients(error2,model.Ws[0])[0].eval(feed_dict=train_s())
                    
                    # grads_and_vars = opt.compute_gradients(error2, var_list=model.var_list)
                    # grad_SSEs = [tf.reduce_sum(var**2) for var in grads_and_vars]
                    # print(sess.run(grads_and_vars,feed_dict=train_s()))
                    # opt.apply_gradients(grads_and_vars).run(feed_dict=train_s())
                    # update=opt.apply_gradients(grads_and_vars)
                    # sess.run((grads_and_vars,update),feed_dict=train_s())
                    # grad_t = sess.run(grads_and_vars, feed_dict=train_s())
                    # sess.run(update, feed_dict=train_s())
                    # print(grads_and_vars)
                    # tmp_grads = grads_and_vars[0][0].eval()
                    # print(tmp_grads)
                    # grads_norm=np.linalg.norm(tmp_grads)
                    # print(grads_norm)
                    # if grads_norm<0.01:
                    #     pass
                        # print(t_idx)
                        # break

                # baseline_test_accuracy = accuracy2.eval(feed_dict=valid_s())
                # print(tf.get_default_graph().as_graph_def())
                train_error = error2.eval(feed_dict=train_s())
                test_error = error.eval(feed_dict=test_s())
                valid_error = error.eval(feed_dict=valid_s())
                # print(train_error)
                # writer = tf.summary.FileWriter('./graphs2', graph)
                # print('baseline_test_accuracy',baseline_test_accuracy)
                # return 1-baseline_test_accuracy
                # return test_error, train_error, test_s, model.Ws[0].eval(),model.bs[0].eval()
                return valid_error, train_error, test_error

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
            # test_error = error.eval(feed_dict=valid_s())
            # print('baseline_test_accuracy',baseline_test_accuracy)
            return 1-baseline_test_accuracy
            # return test_error

    def _get_projector(self, R, N_ex): # !
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

    def extend_vector(self, vector, extend_szie):
        part_size = int(np.ceil(extend_szie/vector.size))
        tmp_vector=np.ones(extend_szie,dtype='float32')
        for p_idx in range(vector.size):
                if p_idx == vector.size-1:

                    tmp_vector[part_size*p_idx:]=np.ones(extend_szie-(vector.size-1)*part_size,dtype='float32')*vector[p_idx]
                else:
                    # print(part_size*p_idx)
                    # print(part_size*(p_idx+1)-1)
                    # print(tmp_part_lambdak[p_idx])
                    tmp_vector[part_size*p_idx:(part_size*(p_idx+1))]=np.ones(part_size,dtype='float32')*vector[p_idx]

        return tmp_vector