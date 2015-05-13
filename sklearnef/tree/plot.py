#!/usr/bin/python

import random
import scipy.stats
import numpy as np
from tree import UnSupervisedDecisionTreeClassifier

distance = 10
sigma = 0.4
n_samples = 1000
n_features = 2
n_clusters = 5
means = [[i * distance] * n_features for i in range(n_clusters)]
means = []
for i in range(n_clusters):
    means.append([])
    for j in range(n_features):
        means[i].append(random.randint(0, distance))
cov = np.diag([sigma] * n_features)
cov[np.triu_indices(n_features, 1)] = sigma/2
cov[np.tril_indices(n_features, -1)] = sigma/2

#!TODO: Should I fix the random state?
X_train = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])
X_test = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])
y_test = np.sum([scipy.stats.multivariate_normal.pdf(X_test, mean, cov) for mean in means], 0)
y_test_log = np.sum([scipy.stats.multivariate_normal.logpdf(X_test, mean, cov) for mean in means], 0)

clf = UnSupervisedDecisionTreeClassifier(random_state=0)
clf.fit(X_train)
prob_predict = clf.predict_proba(X_test)
    
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

x = X_test[:,0]
y = X_test[:,1]

z_min = min(prob_predict.min(), y_test.min())
z_max = max(prob_predict.max(), y_test.max())

# first plot
plt.subplot(2, 1, 1)
xi = np.linspace(x.min(),x.max(),100)
yi = np.linspace(y.min(),y.max(),100)
zi = griddata(x,y,y_test,xi,yi,interp='linear')

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.PiYG)
plt.colorbar() # draw colorbar

#plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
plt.title('GT')

# second plot
plt.subplot(2, 1, 2)
xi = np.linspace(x.min(),x.max(),100)
yi = np.linspace(y.min(),y.max(),100)
zi = griddata(x,y,prob_predict,xi,yi,interp='linear')

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.PiYG)
plt.colorbar() # draw colorbar

#plt.scatter(x,y,marker='o',c='b',s=5)
plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
plt.title('learned')

plt.show()
