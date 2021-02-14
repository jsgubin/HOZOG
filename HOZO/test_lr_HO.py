from hozo import LogisticRegressionHO

# load some data
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
import time

# get a training set and test set
data_train = datasets.fetch_20newsgroups_vectorized(subset='train')
data_test = datasets.fetch_20newsgroups_vectorized(subset='test')

X_train = data_train.data
X_test = data_test.data
y_train = data_train.target
y_test = data_test.target

# binarize labels
y_train[data_train.target < 10] = -1
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = -1
y_test[data_test.target >= 10] = 1

# optimize model parameters and hyperparameters jointly
# using HOAG
start_time = time.time()
clf = LogisticRegressionHO(max_iter=50, eta=50, mu=1)
clf.fit_CV(X_train, y_train, X_test, y_test, folds=5)
total_time = time.time()-start_time
print('HOZO time:',total_time)

print('Regularization chosen by HOZO: lambda=%s' % (clf.lambdak[0]))

# range for regularization parameters
# lambdas = np.linspace(-25, 10, 40)

# def cost_func(a):
#     clf = linear_model.LogisticRegression(
#         solver='lbfgs',
#         C=np.exp(-a), fit_intercept=False, 
#         tol=1e-22, max_iter=500)

#     clf.fit(X_train, y_train)
#     #cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0.)
#     cost = clf.score(X_test, y_test)
#     return cost

# scores = [cost_func(a) for a in lambdas]

# # make the plot bigger than default
# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['font.size'] = 20

# # plot the scores
# plt.plot(lambdas, scores, lw=3, label='cross-validation error')
# plt.xlabel(r'$\alpha$', fontsize=40)

# # plot HOAG value of alpha
# plt.plot((clf.lambdak[0], clf.lambdak[0]), (0,1), c='k', linestyle='--', 
#          label=r'value of regularization ($\alpha$) found by HOZO')
# plt.legend(fontsize=20)
# plt.grid()
# plt.show()

# make the plot bigger than default
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['font.size'] = 20

plt.xlabel('time', fontsize=40)
plt.plot(clf.times, clf.losses, lw=3, label='cross-validation error')
plt.legend(fontsize=20)
plt.grid()
plt.show()