#!/usr/bin/env python

import scipy.stats
import numpy as np
from tree import UnSupervisedDecisionTreeClassifier
import matplotlib.pyplot as plt

# ----- Constants -----
distance = 10
sigma = 0.4
resolution = .5
n_samples = 1000
n_features = 2
n_clusters = 2
np.random.seed(0)

# ----- Clusters -----
means = [[i * distance] * n_features for i in range(n_clusters)]
means = []
for i in range(n_clusters):
    means.append([])
    for j in range(n_features):
        means[i].append(np.random.randint(0, distance))
cov = np.diag([np.random.random() * sigma for _ in range(n_features)])
n_tri_elements = (n_features * (n_features - 1)) / 2
cov[np.triu_indices(n_features, 1)] = [np.random.random() * sigma/2 for _ in range(n_tri_elements)]
cov[np.tril_indices(n_features, -1)] = [np.random.random() * sigma/2 for _ in range(n_tri_elements)]

# ----- Sample train set -----
X_train = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])

# ----- Grid -----
x_lower = X_train[:,0].min() - 2 * sigma
x_upper = X_train[:,0].max() + 2 * sigma
y_lower = X_train[:,1].min() - 2 * sigma
y_upper = X_train[:,1].max() + 2 * sigma
grid = np.mgrid[x_lower:x_upper:resolution,y_lower:y_upper:resolution]


# ----- Training -----
clf = UnSupervisedDecisionTreeClassifier(random_state=0)
clf.fit(X_train)

# ----- Prediction -----
X_test_pred = np.rollaxis(grid, 0, 3).reshape((np.product(grid.shape[1:]), grid.shape[0]))
prob_predict = clf.predict_proba(X_test_pred)

# ----- Ground truth -----
X_test_gt = np.rollaxis(grid, 0, 3)
prob_gt = np.sum([scipy.stats.multivariate_normal.pdf(X_test_gt, mean, cov) for mean in means], 0)

# ----- Plotting -----
x, y = grid
x = np.unique(x)
y = np.unique(y)

# first plot: gt
plt.subplot(2, 1, 1)
CS = plt.contour(x,y,prob_gt.T,15,linewidths=0.5,colors='k')
CS = plt.contourf(x,y,prob_gt.T,15,cmap=plt.cm.PiYG)
plt.colorbar()

plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
plt.title('GT')

# second plot: prediction
plt.subplot(2, 1, 2)
CS = plt.contour(x,y,prob_predict.reshape((x.size,y.size)).T,15,linewidths=0.5,colors='k')
CS = plt.contourf(x,y,prob_predict.reshape((x.size,y.size)).T,15,cmap=plt.cm.PiYG)
plt.colorbar()

plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))
plt.title('Prediction')

print plt

plt.show()


### OLDER ###
#X_test = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean in means])
#y_test = np.sum([scipy.stats.multivariate_normal.pdf(X_test, mean, cov) for mean in means], 0)
#y_test_log = np.sum([scipy.stats.multivariate_normal.logpdf(X_test, mean, cov) for mean in means], 0)

#x = X_test[:,0]
#y = X_test[:,1]

#z_min = min(prob_predict.min(), y_test.min())
#z_max = max(prob_predict.max(), y_test.max())

# first plot
#plt.subplot(2, 1, 1)
#xi = np.linspace(x.min(),x.max(),100)
#yi = np.linspace(y.min(),y.max(),100)
#zi = griddata(x,y,y_test,xi,yi,interp='linear')

#CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.PiYG)
#plt.colorbar() # draw colorbar

#plt.scatter(x,y,marker='o',c='b',s=5)
#plt.xlim(min(x),max(x))
#plt.ylim(min(y),max(y))
#plt.title('GT')

# second plot
#plt.subplot(2, 1, 2)
#xi = np.linspace(x.min(),x.max(),100)
#yi = np.linspace(y.min(),y.max(),100)
#zi = griddata(x,y,prob_predict,xi,yi,interp='linear')

#CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.PiYG)
#plt.colorbar() # draw colorbar

#plt.scatter(x,y,marker='o',c='b',s=5)
#plt.xlim(min(x),max(x))
#plt.ylim(min(y),max(y))
#plt.title('learned')

#plt.show()
