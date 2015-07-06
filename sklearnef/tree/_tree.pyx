# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change!

from libc.string cimport memcpy, memset
from libc.math cimport log

cimport numpy as np
import numpy as np
np.import_array()

from sklearn.tree._tree cimport Criterion, Splitter, SplitRecord
from sklearnef.tree._diffentropy cimport Diffentropy
cimport sklearn.tree._tree
import sklearn.tree._tree

cdef extern class sklearn.tree._tree.ClassificationCriterion(Criterion):
    cdef SIZE_t* n_classes
    cdef SIZE_t label_count_stride
    cdef double* label_count_left
    cdef double* label_count_right
    cdef double* label_count_total
cdef extern class sklearn.tree._tree.Entropy(ClassificationCriterion):
    pass

cdef extern class sklearn.tree._tree.BaseDenseSplitter(Splitter):
    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_fx_stride
cdef extern class sklearn.tree._tree.BestSplitter(BaseDenseSplitter):
    pass

# =============================================================================
# Types and constants
# =============================================================================
# Any differential entropy value nearing this number is considered dense enough
# and no new split are considered for the associated node (i.e. it will be
# declared a leaf node). 
cdef double ENTROPY_SHIFT = -log(1e-6)

# Infinity definition as used in sklearn.tree._tree
cdef double INFINITY = np.inf

# =============================================================================
# Criterion
# =============================================================================

cdef class UnSupervisedClassificationCriterion(Criterion):
    """Criterion for un-supervised classification using differential entropy."""
    # note: sample weights can not be incorporated, assuming equal weight!
    # but: it could be introduced using weighted sample sums in the impurity_improvement method

    def __cinit__(self, SIZE_t n_samples, SIZE_t n_features, DTYPE_t min_improvement):
        # Default values
        self.X = NULL # all training samples
        self.X_stride = 0 # stride (i.e. feature length * feature_dtype_length)
        self.sample_weight = NULL # sample weight in 1-D double array

        self.samples = NULL # sample ids, ordered by splitter between start and end
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples # X.shape[0]
        self.n_features = n_features # X.shape[1]
        self.n_node_samples = 0 # samples in split to consider
        self.weighted_n_node_samples = 0.0 # total weight of all node samples
        self.weighted_n_left = 0.0 # weight of all samples in left child node
        self.weighted_n_right = 0.0 # weight of all samples in right child node
        
        self.min_improvement = min_improvement
        
        self.covl = Diffentropy(n_features) # left updateable cov for diffentropy calculation
        self.covr = Diffentropy(n_features) # right updateable cov for diffentropy calculation
        
        # The number of 'effective' prior observations (default = 0).
        #self.effprior = 3.
        # The variance of the effective observations (default = 900).
        #self.effpriorvar = 900.
    
    def __dealloc__(self):
        """Destructor."""
        pass
        
    def __reduce__(self): # arguemnts to __cinit__
        return (UnSupervisedClassificationCriterion,
                (self.n_samples,
                 self.n_features,
                 self.min_improvement),
                 self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass    

    cdef void init2(self, DTYPE_t* X, SIZE_t X_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
           
           Note that the signature of this init2() does not
           correspond to the parent classes (Criterion) init()
           signature and therefore can only be used by a dedicated
           splitter class."""
        # Initialize fields
        self.X = X
        self.X_stride = X_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        # initialize weighted_n_node_samples
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double weighted_n_node_samples = 0.0
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        
        #!TODO: What is with zero-weight samples? Are they already removed from "samples"
        #       or would I have to take care of this here? After all, they should not figure
        #       in the entropy computation.
        #       Note: Splitter.init() removes zero-weighted samples from the "samples" list, i.e. they do not appear hear at all.
        #       This is anyway a problem, as the weight figure in the information gain, but not
        #       directly in the entropy computation -> could this be a problem?
        
        if sample_weight == NULL:
            weighted_n_node_samples = 1.0 * n_node_samples
        else:
            for p in range(start, end):
                i = samples[p]
                weighted_n_node_samples += sample_weight[i]

        self.weighted_n_node_samples = weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef DTYPE_t* X = self.X
        
        self.pos = self.start
        
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        
        self.covl.reset()
        self.covr.reset()

        for i in range(start, end):
            self.covr.update_add(X + samples[i] * X_stride)
            
    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child."""
        cdef DTYPE_t* X = self.X
        cdef SIZE_t X_stride = self.X_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t diff_w = 0.0        
        
        for p in range(pos, new_pos):
            i = samples[p]
            self.covl.update_add(X + i * X_stride)
            self.covr.update_sub(X + i * X_stride)
            if sample_weight == NULL:
                diff_w = 1.0 * (new_pos - pos)
            else:
                diff_w += sample_weight[i]
                
        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w
        
        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        """Compute the impurity of the current node."""
        # NOTE: The builders call this function only once for the first node.
        #       Therefore it is possible to compute a dedicated cov here for
        #       single use.

        cdef double entropy = 0.0
        
        self.covr.compute_covariance_matrix()
        entropy = self.covr.logdet()

        return entropy + ENTROPY_SHIFT

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        self.covl.compute_covariance_matrix()
        impurity_left[0] = self.covl.logdet() + ENTROPY_SHIFT
        self.covr.compute_covariance_matrix()
        impurity_right[0] = self.covr.logdet() + ENTROPY_SHIFT

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        #!TODO: Here the variables of the fitted multivariate Gaussian distribution
        #       represented by this leaf node should be written into the given memory.
        # I must figure out    1. how many variables this will be and
        #                      2. how to ensure that the tree provides the required size (n_classes/output * n_outputs * double)
        
        # memory space provide:
        #     [n_outputs, max_n_classes] of double
        # original purpose:
        #     Contains the constant prediction value of each node.
        # I require:
        #     n_features + n_features^2 + 1 of double
        #     mu           cov            frac/Z
        
        # Computation order:
        #    Here I can calculate the cov, mu and frac
        #    Afterwards, I'll have to come back to the tree and calculate
        #    partition funtion Z and replace frac by  frac/Z for each node
        
        # !TODO: replace with nogil version
        # !TODO: Very crude version!
        # convert memory block to numpy array
        
        cdef:
            SIZE_t n_features = self.n_features
            DOUBLE_t weighted_n_node_samples = self.weighted_n_node_samples
            DOUBLE_t weighted_n_samples = self.weighted_n_samples
            DOUBLE_t frac = 0.
            DOUBLE_t* cov = NULL
            DOUBLE_t* mu = NULL
        
        # reset, such that all samples in right sided cov containes
        self.reset()
         
        # trigger cov computation
        self.covr.compute_covariance_matrix()
         
        # fetch cov, mean and fraction of samples in node
        cov = self.covr.cov
        mu = self.covr.__mean
        frac = weighted_n_node_samples / weighted_n_samples
         
        # copy data to target memory
        memcpy(dest, &frac, sizeof(DOUBLE_t))
        dest += 1
        upper_to_matrix(dest, cov, n_features) #!TODO: this can actually be stored in upper triangular format
        dest += n_features * n_features
        memcpy(dest, mu, n_features * sizeof(DOUBLE_t))
        
    cdef double impurity_improvement(self, double impurity) nogil:
        """Weighted impurity improvement, i.e.

           N_t / N * (impurity - N_t_L / N_t * left impurity
                               - N_t_L / N_t * right impurity),

           where N is the total number of samples, N_t is the number of samples
           in the current node, N_t_L is the number of samples in the left
           child and N_t_R is the number of samples in the right child.
           
           Extended version that catches any improvement < min_improvement
           and returns -INFINITY instead to invalidate the current split."""
        cdef double improvement
        cdef DTYPE_t min_improvement

        improvement = Criterion.impurity_improvement(self, impurity)
        min_improvement = self.min_improvement

        return improvement if improvement >= min_improvement else -INFINITY

cdef inline void upper_to_matrix(DOUBLE_t* X, DOUBLE_t* Y, SIZE_t length) nogil:
    "Convert the upper triangular matrix Y to full matrix X assuming symmetry."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    # first copy existing elements to upper
    for p1 in range(length):
        for p2 in range(p1, length):
            X[p2 + p1 * length] = Y[0]
            Y += 1
    
    # copy triangular symmetric elements from upper to lower (excluding diagonal)
    for p1 in range(1, length):
        for p2 in range(0, p1):
            X[p2 + p1 * length] = X[p1 + p2 * length]

cdef class SemiSupervisedClassificationCriterion(Criterion):
    """Semi-supervised criteria that combines `LabeledOnlyEntropy` and
    `UnSupervisedClassificationCriterion` into one balanced quality measure.
    """ 
    
    def __cinit__(self, SIZE_t n_samples, SIZE_t n_features, DTYPE_t supervised_weight, 
                  SIZE_t n_outputs, np.ndarray[SIZE_t, ndim=1] n_classes):
        self.supervised_weight = supervised_weight
        self.criterion_unsupervised = UnSupervisedClassificationCriterion(n_samples, n_features, min_improvement = 0.0)
        self.criterion_supervised = LabeledOnlyEntropy(n_outputs, n_classes)
        
    cdef void init3(self, DTYPE_t* X, SIZE_t X_stride, DOUBLE_t* y, SIZE_t y_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
           
           Note that the signature of this init3() does not
           correspond to the parent classes (Criterion) init()
           signature and therefore can only be used by a dedicated
           splitter class."""
        self.criterion_unsupervised.init2(X, X_stride, sample_weight, weighted_n_samples,
                                          samples, start, end)
        self.criterion_supervised.init(y, y_stride, sample_weight, weighted_n_samples,
                                       samples, start, end)
        
    cdef void reset(self) nogil:
        self.criterion_unsupervised.reset()
        self.criterion_supervised.reset()
        
    cdef void update(self, SIZE_t new_pos) nogil:
        self.criterion_unsupervised.update(new_pos)
        self.criterion_supervised.update(new_pos)
        
    cdef double node_impurity(self) nogil:
        cdef double uimp, simp
        cdef DTYPE_t supervised_weight
        
        supervised_weight = self.supervised_weight
        uimp = self.criterion_unsupervised.node_impurity()
        simp = self.criterion_supervised.node_impurity()
        
        return (1. - supervised_weight) * uimp + supervised_weight * simp
    
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef DTYPE_t supervised_weight
        cdef double uimp_impurity_left
        cdef double uimp_impurity_right
        cdef double simp_impurity_left
        cdef double simp_impurity_right
        
        supervised_weight = self.supervised_weight
        self.criterion_unsupervised.children_impurity(&uimp_impurity_left, &uimp_impurity_right)
        self.criterion_supervised.children_impurity(&simp_impurity_left, &simp_impurity_right)
        
        impurity_left[0] = (1. - supervised_weight) * uimp_impurity_left + supervised_weight * simp_impurity_left
        impurity_right[0] = (1. - supervised_weight) * uimp_impurity_right + supervised_weight * simp_impurity_right
        
    cdef void node_value(self, double* dest) nogil:
        self.criterion_unsupervised.node_value(dest)
        self.criterion_supervised.node_value(dest)
        
    cdef double impurity_improvement(self, double impurity) nogil:
        cdef double uimp, simp
        cdef DTYPE_t supervised_weight
        
        supervised_weight = self.supervised_weight
        uimp = self.criterion_unsupervised.impurity_improvement(impurity)
        simp = self.criterion_supervised.impurity_improvement(impurity)
        
        return (1. - supervised_weight) * uimp + supervised_weight * simp
    
    #!TODO: Is the unsupervised the right criterion to ask these values from?
    property weighted_n_node_samples:
        def __get__(self):
            return self.criterion_unsupervised.weighted_n_node_samples
        
    property weighted_n_left:
        def __get__(self):
            return self.criterion_unsupervised.weighted_n_left
    
    property weighted_n_right:
        def __get__(self):
            return self.criterion_unsupervised.weighted_n_right
    
cdef class LabeledOnlyEntropy(Entropy):
    """Cross Entropy impurity criteria applied to labeled samples only."""
 
    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride,
                   DOUBLE_t* sample_weight, double weighted_n_samples,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.0
 
        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
 
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0
 
        for k in range(n_outputs):
            memset(label_count_total + offset, 0,
                   n_classes[k] * sizeof(double))
            offset += label_count_stride
 
        for p in range(start, end):
            i = samples[p]
 
            if sample_weight != NULL:
                w = sample_weight[i]
 
            c = <SIZE_t> y[i * y_stride]
            if c > 0: # don't consider unlabeled samples; assuming that sample is either for all outputs unlabeled or for all labeled
                label_count_total[k * label_count_stride + c - 1] += w
                 
                for k in range(1, n_outputs):
                    c = <SIZE_t> y[i * y_stride + k]
                    label_count_total[k * label_count_stride + c - 1] += w
                 
                weighted_n_node_samples += w
        self.weighted_n_node_samples = weighted_n_node_samples
 
        # Reset to pos=start
        self.reset()
 
    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight
 
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
 
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right
 
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t c = 0
        cdef SIZE_t k = 0
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t diff_w = 0.0
 
        # Note: We assume start <= pos < new_pos <= end
 
        for p in range(pos, new_pos):
            i = samples[p]
 
            if sample_weight != NULL:
                w = sample_weight[i]
 
            c = <SIZE_t> y[i * y_stride]
            if c > 0: # don't consider unlabeled samples; assuming that sample is either for all outputs unlabeled or for all labeled
                label_index = k * label_count_stride + c - 1
                label_count_left[label_index] += w
                label_count_right[label_index] -= w
                 
                for k in range(1, n_outputs):
                    c = <SIZE_t> y[i * y_stride + k]
                    label_index = k * label_count_stride + c - 1
                    label_count_left[label_index] += w
                    label_count_right[label_index] -= w
     
                diff_w += w
 
        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w
 
        self.pos = new_pos   
 
# =============================================================================
# Splitter
# =============================================================================
 
cdef class UnSupervisedBestSplitter(BestSplitter):
    """Splitter for finding the best split on un-labelled data."""
    cdef UnSupervisedClassificationCriterion criterion_real
     
    def __cinit__(self, UnSupervisedClassificationCriterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        self.criterion_real = criterion
     
    def __reduce__(self):
        return (UnSupervisedBestSplitter, (self.criterion,
                                           self.max_features,
                                           self.min_samples_leaf,
                                           self.min_weight_leaf,
                                           self.random_state), self.__getstate__())    
 
    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end]."""
        self.start = start
        self.end = end
 
        self.criterion_real.init2(self.X,
                                  self.X_sample_stride,
                                  self.sample_weight,
                                  self.weighted_n_samples,
                                  self.samples,
                                  start,
                                  end)
 
        weighted_n_node_samples[0] = self.criterion_real.weighted_n_node_samples

cdef class SemiSupervisedBestSplitter(BestSplitter):
    """Splitter for finding the best split on partially labelled data."""
    cdef SemiSupervisedClassificationCriterion criterion_real
     
    def __cinit__(self, SemiSupervisedClassificationCriterion criterion, SIZE_t max_features,
                   SIZE_t min_samples_leaf, double min_weight_leaf,
                   object random_state):
        self.criterion_real = criterion
     
    def __reduce__(self):
        return (SemiSupervisedBestSplitter, (self.criterion,
                                             self.max_features,
                                             self.min_samples_leaf,
                                             self.min_weight_leaf,
                                             self.random_state), self.__getstate__())    
 
    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end]."""
        self.start = start
        self.end = end
 
        self.criterion_real.init3(self.X,
                                  self.X_sample_stride,
                                  self.y,
                                  self.y_stride,
                                  self.sample_weight,
                                  self.weighted_n_samples,
                                  self.samples,
                                  start,
                                  end)
 
        weighted_n_node_samples[0] = self.criterion_real.weighted_n_node_samples
