import numpy as np
import time
import hozo
from pathos.multiprocessing import ProcessingPool

class HOZO():

    def __init__(self, model, max_iter=200, mu=1e-3, 
        q=5, eta=5, rv_dist='UnitSphere', process_num=None):

        self.model = model
        self.max_iter = max_iter
        self.mu = mu
        self.q = q
        self.eta = eta
        self.time = []
        self.losses = []
        self.lambdas = []
        self.tr_loss = []
        self.t_loss = []
        if not process_num:
            # current loss and q new results
            self.process_num = 1 + q
            print('self.process_num', self.process_num)

        if(rv_dist == 'UnitBall'):
            self.RV_Gen = self.Draw_UnitBall
        elif(rv_dist == 'UnitSphere'):
            self.RV_Gen = self.Draw_UnitSphere
        else:
            print('Please specify a valid distribution for random perturbation')

    def fit(self, lmd0, projection=None, name=None, **kw):
        time_start = time.time()

        # gaps = []
        # zog = []
        
        eta = self.eta

        best_loss = float('inf')
        best_lambda = lmd0
        lambdak = lmd0

        print('---HOZO Start!')

        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            

            zo_gradient, v_loss, avg_gap = \
        self.full_zo_gradient_estimation(lmd=lambdak, **kw)

            lambdak = lambdak - eta *  zo_gradient
            if projection:
                lambdak = projection(lambdak)            

            if(v_loss < best_loss):
                best_loss = v_loss
                best_lambda = lambdak
 
            if(iter%20 == 0):
                self.losses.append(v_loss)
                self.time.append(time.time()-time_start)
                print('---Iteration: ', iter)
                print('losses', self.losses)
                print('time',self.time)
            # zog.append(zo_gradient)
            # gaps.append(avg_gap)
            # print('gaps',gaps)

        self.lambdak = best_lambda
        return self

    # TODO
    def fit_ADAM(self, lmd0, beta1=.9, beta2=.999, 
        eps=1e-6, projection=None, name=None, **kw):
    
        time_start = time.time()

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

    # TODO
    def fit_NAG(self, data, part_lmd, saver=None, T=2000, lr=.1, discount=.7, projection=None):
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
            
            new_loss = 1e10
            
            while new_loss>old_loss+0.002:
                part_pre_grad2 = part_pre_grad
                pre_grad2 = pre_grad

                part_lambda_future = part_lmd - self.eta*discount*part_pre_grad2
                lambda_future = lambdak - self.eta*discount*pre_grad2

                part_lambda_future = projection(part_lambda_future)
                lambda_future = projection(lambda_future)

                loss, zo_gradient, part_zo_gradient = \
            self.part_full_zo_gradient_estimation_oldloss(old_loss=old_loss, part_lmd=part_lambda_future, saver=saver, model=model, y=y, data=data, T=T, lr=lr, lmd=lambda_future, projection=projection)
                
                pre_grad2 = pre_grad2*discount+zo_gradient
                part_pre_grad2 = part_pre_grad2*discount+part_zo_gradient

                part_lmd2 = part_lmd - self.eta * part_pre_grad2
                lambdak2 = lambdak - self.eta * pre_grad2

                print('part_lmd2',part_lmd2)
                print('lambdak2',lambdak2)

                part_lmd2 = projection(part_lmd2)
                lambdak2 = projection(lambdak2)
                new_loss = self.cost_func(saver, model, y, data, T, lr, lambdak2, name='New loss')
                print('new_loss',new_loss)
            
            part_pre_grad = part_pre_grad2
            pre_grad = pre_grad2
            part_lmd = part_lmd2
            lambdak = lambdak2
            old_loss = new_loss
            if(iter%1 == 0):
                print('Iteration Index: ', iter)
                print('Current loss:',loss)
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

    def full_zo_gradient_estimation(self, lmd, name='zo gradient', **kw):
        model = self.model
        mu = self.mu
        q = self.q
        process_num = self.process_num

        grad_avg = np.zeros(lmd.size, dtype='float32')
        lmds = []
        u_rands = []
        # For current loss.
        lmds.append(lmd)

        for q_idx in range(q):
            u_rand = self.RV_Gen(lmd.size)
            tmp_lambdak = lmd + mu * u_rand
            lmds.append(tmp_lambdak)
            u_rands.append(u_rand)

        pool = ProcessingPool(nodes=process_num)
        init_model_dict = kw['init_model_dict']
        data_dict = kw['data_dict']

        losses = pool.map(lambda lmd:model(lmd, **init_model_dict).train_valid(**data_dict), np.array(lmds))

        # print(losses)
        # print(losses[1:])
        avg_gap = np.mean(losses[1:]-losses[0])
        # print(avg_gap)
        grad_avg=np.dot((losses[1:]-losses[0]),u_rands)
        # print(grad_avg)

        # ZOG, loss, avg_gap
        return (lmd.size/mu)*(grad_avg/q), losses[0], avg_gap/q