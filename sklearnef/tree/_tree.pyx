# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change!

from libc.string cimport memcpy, memset

cimport numpy as np
np.import_array()

from sklearn.tree._tree cimport Criterion, Splitter, SplitRecord
from sklearnef.tree._diffentropy cimport Diffentropy

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

# =============================================================================
# Criterion
# =============================================================================

cdef class UnSupervisedClassificationCriterion(Criterion):
    """Criterion for un-supervised classification using differential entropy."""
    # note: sample weights can not be incorporated, assuming equal weight!
    # but: it could be introduced using weighted sample sums in the impurity_improvement method

    def __cinit__(self, SIZE_t n_samples, SIZE_t n_features):
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
                 self.n_features),
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
        
        return entropy

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        self.covl.compute_covariance_matrix()
        impurity_left[0] = self.covl.logdet()
        self.covr.compute_covariance_matrix()
        impurity_right[0] = self.covr.logdet()

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

# cdef inline SIZE_t upper_n_elements(SIZE_t n) nogil:
#     "The number of (diagonal including) elements of an upper triangular nxn matrix."
#     return (n * n + n) / 2
 
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
                label_index = k * label_count_stride + c
                label_count_left[label_index] += w
                label_count_right[label_index] -= w
                 
                for k in range(1, n_outputs):
                    label_index = (k * label_count_stride +
                                   <SIZE_t> y[i * y_stride + k])
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

