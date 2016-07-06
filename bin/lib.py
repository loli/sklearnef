"""
Shared library for the sample generators.
r0.1.0, 2016-06-06
Oskar Maier <oskar.maier@googlemail.com>
"""

# third-party modules
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# constants
N_FEATURES = 2

# code
def generate_clusters(max_area, n_clusters, sigma):
    """Generate n_cluster random clusters with random variance sigma distributed over an area of max_area^n_features."""
    # ----- Define gaussian distributions / clusters -----
    means = []
    for _ in range(n_clusters):
        means.append([np.random.randint(0, max_area) for _ in range(N_FEATURES)])
    covs = []
    for _ in range(n_clusters):
        cov = np.diag([(np.random.random() + .5) * sigma for _ in range(N_FEATURES)])
        n_tri_elements = (N_FEATURES * (N_FEATURES - 1)) / 2
        cov[np.triu_indices(N_FEATURES, 1)] = [(np.random.random() + .5) * sigma/2 for _ in range(n_tri_elements)]
        cov[np.tril_indices(N_FEATURES, -1)] = [(np.random.random() + .5) * sigma/2 for _ in range(n_tri_elements)]
        covs.append(cov)
    return means, covs

def sample_data(means, covs, n_samples):
    """Sample training and testing data."""
    # ----- Sample train set -----
    X_train_unlabelled = np.concatenate([scipy.stats.multivariate_normal.rvs(mean, cov, n_samples) for mean, cov in zip(means, covs)]).astype(np.float32)
    y_train_unlabelled = np.full(X_train_unlabelled.shape[0], -1)
    y_train_gt = np.concatenate([[c] * n_samples for c in np.arange(len(means))], 0)
    X_train_labelled = np.asarray(means).astype(np.float32)
    y_train_labelled = np.arange(len(means)) 

    X_train = np.concatenate((X_train_unlabelled, X_train_labelled), 0)
    y_train = np.concatenate((y_train_unlabelled, y_train_labelled), 0)
    
    return  (X_train, X_train_unlabelled, X_train_labelled),\
            (y_train, y_train_unlabelled, y_train_labelled),\
            y_train_gt
            
def scale_data(fit_data, transform_data):
    """Scale the data with a standard scaler."""
    ss = StandardScaler().fit(fit_data)
    return (ss.transform(data) for data in transform_data)

def generate_grid(data, sigma, resolution):
    """Generate a sample grid for the data."""
    x_lower = data[:,0].min() - 2 * sigma / np.std(data[:,0])
    x_upper = data[:,0].max() + 2 * sigma / np.std(data[:,0])
    y_lower = data[:,1].min() - 2 * sigma / np.std(data[:,1])
    y_upper = data[:,1].max() + 2 * sigma / np.std(data[:,1])
    grid = np.mgrid[x_lower:x_upper:(x_upper-x_lower)/float(resolution),y_lower:y_upper:(y_upper-y_lower)/float(resolution)]
    return grid

def draw_split_lines(clf, x, y):
    """Draw the trees split lines into the current image."""
    info = clf.parse_tree_leaves()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    hlines = ([], [], [])
    vlines = ([], [], [])
    for node in info:
        if node is not None: # leave node
            for pos in node.unscaled_range[0]: # xrange
                if not np.isinf(pos):
                    xliml, xlimr = node.unscaled_range[1]
                    vlines[0].append(pos)
                    vlines[1].append(ymin if np.isinf(xliml) else xliml)
                    vlines[2].append(ymax if np.isinf(xlimr) else xlimr)
            for pos in node.unscaled_range[1]: # xrange
                if not np.isinf(pos):
                    yliml, ylimr = node.unscaled_range[0]
                    hlines[0].append(pos)
                    hlines[1].append(xmin if np.isinf(yliml) else yliml)
                    hlines[2].append(xmax if np.isinf(ylimr) else ylimr)
    plt.hlines(hlines[0], hlines[1], hlines[2], colors='blue', linestyles='dotted')
    plt.vlines(vlines[0], vlines[1], vlines[2], colors='blue', linestyles='dotted')
    
def make_sklearn_dataset(dataset_name, n_samples):
    # create dataset
    if 'circles_distant' == dataset_name: # labels=3, seed=1, n-samples=1000, max-depth=4 OR labels=4, seed=1, n-samples=1000, max-depth=4
        dataset = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif 'moons' == dataset_name: # labels=2, seed=13, n-samples=500, max-depth=4 OR labels=1, seed=27, n-samples=500, max-depth=4
        dataset = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif 'blobs' == dataset_name: # labels=1, seed=0, n-samples=100, max-depth=3 
        dataset = datasets.make_blobs(n_samples=n_samples, random_state=8)
    elif 'circles_near' == dataset_name: # labels = 20, seed=0, n-samples=2000, max-depth=5
        dataset = datasets.make_circles(n_samples=n_samples, noise=.05)
    elif 's_curve' == dataset_name: # labels=10, seed=35, n-samples=2500, max-depth=7
        scurve1 = datasets.make_s_curve(n_samples=n_samples // 2, noise=.05)
        scurve1 = np.vstack((scurve1[0][:, 0], scurve1[0][:, 2])).T
        scurve2 = datasets.make_s_curve(n_samples=n_samples // 2, noise=.05)
        scurve2 = np.vstack((scurve2[0][:, 0], scurve2[0][:, 2])).T + [.5, .5] # offset
        dataset = np.concatenate((scurve1, scurve2), 0), \
                  np.concatenate((np.asarray([0] * scurve1.shape[0]),
                                  np.asarray([1] * scurve2.shape[0])), 0)
    elif 'swiss_roll' == dataset_name: # labels = 10, seed = 35, n-samples=2500, max-depth=5
        sroll1 = datasets.make_swiss_roll(n_samples=n_samples // 2, noise=.05)
        sroll1 = np.vstack((sroll1[0][:,0], sroll1[0][:,2])).T
        sroll2 = datasets.make_swiss_roll(n_samples=n_samples // 2, noise=.05)
        sroll2 = np.vstack((sroll2[0][:,0], sroll2[0][:,2])).T * 0.75 # shrink
        dataset = np.concatenate((sroll1, sroll2), 0), \
                  np.concatenate((np.asarray([0] * sroll1.shape[0]),
                                  np.asarray([1] * sroll2.shape[0])), 0)
                  
    return dataset

def plot_gt(X, x, y, args):
    plt.scatter(X[:, 0], X[:, 1], c='w', alpha=.3, edgecolors='none')
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('GT: {}'.format(args.dataset))
    
def plot_density(prob_predict, x, y, args):
    plt.imshow(prob_predict.reshape((x.size,y.size)).T, extent=[min(x),max(x),min(y),max(y)], interpolation='none', cmap=plt.cm.afmhot, aspect='auto', origin='lower') #'auto'
    plt.colorbar()
    
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.title('Learned density: {}'.format(args.dataset))