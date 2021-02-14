import numpy as np
from scipy import sparse
from sklearn import linear_model
from scipy.special import expit
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from sklearn.model_selection import cross_val_score
import time


class LogisticRegressionHO(linear_model.base.BaseEstimator,
                           linear_model.base.LinearClassifierMixin):

    def __init__(
                 self, lambda0=[1.], inner_tol=1e-4, max_iter=100, mu =0.01, q=2, eta=0.001, rv_dist='UnitSphere'):
        # lambda0: The initialization of the regularization parameter.
    
        # inner_tol: The tolerance of inner optimization.
        
        # max_iter: The max iter for hyperparameter updates.
        
        # mu: The $\mu$ in Eq. (2) of HOZOG paper.
        
        # q: The $m$ in Eq. (2) of HOZOG paper.
        
        # eta: The $\gamma$ in Alg. 1 of HOZOG paper.
        
        # rv_dist: Generate $u$ in Alg. 1 from 'UnitSphere' or 'UnitBall'.

        self.loss = 1e10
        self.lambdak = lambda0
        self.inner_tol = inner_tol
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

    def fit(self, Xtr, ytr, Xt, yt, projection=None):
        # Parameters: 
    
        # Xtr, ytr: Training data.
        # Xt, yt: Test data. 
        # projection: A callable projection function for hyperparameters (Optional).
        
        # Return: 
        # A LogisticRegressionHO instance.

        time_start = time.time()
        if not np.all(np.unique(yt) == np.array([-1, 1])):
            raise ValueError
        
        best_loss = 1e10;
        best_lambda = self.lambdak
        lambdak = self.lambdak
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            loss, zo_gradient = self.full_zo_gradient_estimation(lambdak, Xtr, ytr, Xt, yt)
            print('Current lambdak:',lambdak)
            lambdak = lambdak - self.eta * zo_gradient
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

        self.lambdak = best_lambda
        return self

    def fit_CV(self, Xtr, ytr, Xt, yt, folds=2, projection=None):
        # Parameters: 
    
        # Xtr, ytr: Training data.
        # Xt, yt: Test data. 
        # projection: A callable projection function for hyperparameters (Optional).
        # Set the folds of CV.

        # Return: 
        # A LogisticRegressionHO instance.

        time_start = time.time()
        if not np.all(np.unique(yt) == np.array([-1, 1])):
            raise ValueError

        if folds<1 or not isinstance(folds,int):
            raise ValueError

        best_loss = 1e10;
        best_lambda = self.lambdak
        lambdak = self.lambdak
        for iter in range(self.max_iter):
            self.lambdas.append(lambdak)
            loss, zo_gradient = self.cv_full_zo_gradient_estimation(lambdak, Xtr, ytr, folds)
            print('Current lambdak:',lambdak)
            lambdak = lambdak - self.eta * zo_gradient
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

        print('Test loss:', self.cost_func(lambdak[0], Xtr, ytr, Xt, yt))

        self.lambdak = best_lambda
        return self

    def Draw_UnitBall(self):
        # Draw unit vector $u$ in Alg.1 from a ball
        sample = np.random.uniform(-1.0, 1.0, size=1)
        return sample/np.linalg.norm(sample.flatten())

    def Draw_UnitSphere(self):
        # Draw unit vector $u$ in Alg.1 from a sphere
        sample = np.random.normal(0.0, 1.0, size=1)
        return sample/np.linalg.norm(sample.flatten())

    def full_zo_gradient_estimation(self, lambdak, Xtr, ytr, Xt, yt):
        # Return zo gradients estimated form 0-1 loss.
        mu = self.mu
        q = self.q
        f = self.cost_func(lambdak[0], Xtr, ytr, Xt, yt)
        grad_avg = np.zeros(1)
        for q_idx in range(q):
            u_rand = self.RV_Gen()
            tmp_lambdak = lambdak + mu * u_rand
            f_perturb = self.cost_func(tmp_lambdak[0], Xtr, ytr, Xt, yt)
            grad_avg += (f_perturb-f)*u_rand
        return f,(1/mu)*(grad_avg/q)

    def cv_full_zo_gradient_estimation(self, lambdak, Xtr, ytr, folds):
        # Return zo gradients estimated form CV 0-1 loss.
        mu = self.mu
        q = self.q
        f = self.cv_cost_func(lambdak[0], Xtr, ytr, folds)
        grad_avg = np.zeros(1)
        for q_idx in range(q):
            u_rand = self.RV_Gen()
            tmp_lambdak = lambdak + mu * u_rand
            f_perturb = self.cv_cost_func(tmp_lambdak[0], Xtr, ytr, folds)
            grad_avg += (f_perturb-f)*u_rand
        return f,(1/mu)*(grad_avg/q)

    def cost_func(self, lambdak, Xtr, ytr, Xt, yt):
        # Return 0-1 loss. 
        # Can be modified to output the logistic loss.
        clf = linear_model.LogisticRegression(
            solver='lbfgs',
            C=np.exp(-lambdak), fit_intercept=False, 
            tol=self.inner_tol, max_iter=200)

        clf.fit(Xtr, ytr)
        #cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), Xt, yt, 0.)
        cost = 1 - clf.score(Xt, yt)
        return cost

    def cv_cost_func(self, lambdak, Xtr, ytr, folds):
        # Return CV 0-1 loss. 
        # Can be modified to output the logistic loss.
        clf = linear_model.LogisticRegression(
            solver='lbfgs',
            C=np.exp(-lambdak), fit_intercept=False, 
            tol=self.inner_tol, max_iter=200)

        scores = cross_val_score(clf, Xtr, ytr, cv=folds)
        print('CV scores:', scores)
        #cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), Xt, yt, 0.)
        cost = 1 - np.mean(scores)
        return cost
