# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change!

from libc.stdlib cimport calloc, free, realloc
from libc.string cimport memcpy, memset
from libc.math cimport log as ln

import numpy as np
cimport numpy as np
np.import_array()

from sklearn.tree._tree cimport Criterion, Splitter, SplitRecord

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

cdef extern void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

# =============================================================================
# Types and constants
# =============================================================================

#from sklearn.tree._tree cimport import INFINITY
#cdef extern sklearn.tree._tree.INFINITY
#cimport sklearn.tree._tree
#cdef DTYPE_t MIN_IMPURITY_SPLIT = sklearn.tree._tree.MIN_IMPURITY_SPLIT
#cdef extern DTYPE_t MIN_IMPURITY_SPLIT

cdef double INFINITY = np.inf
cdef DTYPE_t MIN_IMPURITY_SPLIT = 1e-7
# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# =============================================================================
# Criterion
# =============================================================================

cdef class UnSupervisedClassificationCriterion(Criterion):
    """Criterion for un-supervised classification using differential entropy."""
    
    # note: sample weights can not be incorporated, assuming equal weight!
    # but: it could be introduced using weighted sample sums in the impurity_improvement method
    
    # create a copy X' of X
    # always work on memory of X' defined by start - end
    # upon init, place observatins/samples from X into X' according to sample-indices in `samples`, they will be ordered
    # for a split, simply define X_left as the X' from start to pos and X_right as the X' from pos to end
    # for the impurity calculation, compute log(det(\Sigma))...etc on X_left and X_right
    # advanced: try update log(det(\Sigma)) dynamically by moving samples from X_right to X_left

    def __cinit__(self, SIZE_t n_samples, SIZE_t n_features):
        # Default values
        self.X = NULL # all training samples
        self.X_stride = 0 # stride (i.e. feature length * feature_dtype_length)
        self.S = NULL # current node's set of samples, ordered to contain S_left and S_right between start, pos and end
        self.sample_weight = NULL # sample weight in 1-D double array

        self.samples = NULL # sample ids, ordered by splitter between start and end
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples # X.shape[0]
        self.n_features = n_features # X.shape[1]
        self.n_node_samples = 0 # size of S
        self.weighted_n_node_samples = 0.0 # total weight of all node samples (all in S)
        self.weighted_n_left = 0.0 # weight of all samples in S_left
        self.weighted_n_right = 0.0 # weight of all samples in S_right       
        
        # Allocate memory
        self.S = <DTYPE_t*> calloc(n_samples * n_features, sizeof(DTYPE_t)) # same size as X

        # Check for allocation errors
        if self.S == NULL:
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.S)
        
    def __reduce__(self): # arguemnts to __cinit__
        return (UnSupervisedClassificationCriterion,
                (self.n_samples,
                 self.n_features),
                 self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void sortS(self) nogil:
        """Re-sort the sample set S according to sample-id set samples."""
        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* S = self.S
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_node_samples = self.n_node_samples
        
        cdef SIZE_t i = 0
        
        #!TODO: make this a sort, not a re-copy
        for i in range(n_node_samples):
            memcpy(S + (start + i) * X_stride,
                   X + samples[i + start] * X_stride,
                   X_stride * sizeof(DTYPE_t))
            
        #with gil:
        #    print '### SORTS ###'
        

    cdef void init2(self, DTYPE_t* X, SIZE_t X_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
           Also sorts S accordingly.
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

        # sort samples set S
        #self.sortS()

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        
        self.sortS()

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child.""" 
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t diff_w = 0.0        
        
        if sample_weight == NULL:
            diff_w = 1.0 * (new_pos - pos)
        else:
            for p in range(pos, new_pos):
                i = samples[p]
                diff_w += sample_weight[i]
            
        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w
        
        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        """Compute the impurity of the current node."""
        cdef SIZE_t start = self.start
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double entropy = 0.0
        cdef DTYPE_t* S = self.S
        
        #!TODO: Remove the "with gil:" here, when differential_entropy finally nogil function
        with gil:
            entropy = self.differential_entropy(S + start * X_stride, n_node_samples) # src, n_samples
        
        return entropy

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t X_stride = self.X_stride
        cdef DTYPE_t* S = self.S
        
        #!TODO: Remove the "with gil:" here, when differential_entropy finally nogil function
        with gil:
            impurity_left[0] = self.differential_entropy(S + start * X_stride, pos - start)
            impurity_right[0] = self.differential_entropy(S + pos * X_stride, end - pos)

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
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t start = self.start
        cdef DTYPE_t* S = self.S
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef DTYPE_t [:,::1] arr_view
        
        cdef double [:, ::1] mcov
        cdef double [::1] mmu
        
        cdef SIZE_t i = 0
        cdef SIZE_t p = self.n_samples
        
        #with gil:
        #    print '### NODE_VALUE ###'
        #    print 'start/pos/end', self.start, self.pos, self.end
        #    print 'n_node_samples/end-start', self.n_node_samples, self.end - self.start
        #    print 'n_samples', self.n_samples
        #    print 'S:'
        #    for i in range(p):
        #        print i, self.S[i*2], self.S[i*2+1]
              
        with gil:
            arr_view = <DTYPE_t[:n_node_samples,:X_stride]> &S[start * X_stride]
            arr = np.asarray(arr_view).copy().astype(np.float64)
            #print '-----------------'
            #print 'S[0]', S[0]
            #print 'self.S[0]', self.S[0]
            #print 'S[1]', S[1]
            #print 'self.S[1]', self.S[1]
            #print 'start * X_stride', start * X_stride
            #print 'self.S[start * X_stride]', self.S[start * X_stride]
            #print '-----------------'
            #print 'arr.shape', arr.shape
            #print 'arr:'
            #print arr

            cov = np.cov(arr, rowvar=0, ddof=1)
            mu = np.mean(arr, axis=0)
            frac = self.weighted_n_node_samples / self.weighted_n_samples
            
            mcov = cov
            mmu = mu
            
            #print '-----------------'
            #print 'frac:', frac
            #print 'cov:'
            #print cov
            #print 'mu:'
            #print mu
        
        memcpy(dest, &frac, sizeof(double))
        dest += 1
        memcpy(dest, &mcov[0,0], self.n_features * self.n_features * sizeof(double))
        dest += self.n_features * self.n_features
        memcpy(dest, &mmu[0], self.n_features * sizeof(double))

    cdef double differential_entropy(self, DTYPE_t* src, SIZE_t size):
        """Compute the differential entropy (also called continuous- or unsupervised-),
        which is defined as log(det(cov(S))) of a set S of observations identified
        through src and size."""
        cdef SIZE_t X_stride = self.X_stride
        
        #!TODO: employing np.log(MIN_IMPURITY_SPLIT) as a minimal value
        # is insecure, if <10 samples are in the parent node. Min_samples_leaf
        # should be used or the depth restricted or such.
        # The same solution is used below when det < MIN_IMPURITY_SPLIT,
        # which is equally questionable
        
        # skip if set size <= 1
        if size <= 1: return np.log(MIN_IMPURITY_SPLIT)

        # convert memory block to numpy array
        cdef DTYPE_t [:,::1] arr_view = <DTYPE_t[:size,:X_stride]> src
        cdef np.ndarray arr = np.asarray(arr_view).copy()
        
        #!TODO: Place here my dynstatcov and make function nogil
        # compute the differential entropy using numpy
        cov = np.cov(arr, rowvar=0, ddof=1)
        det = np.linalg.det(cov)
        if det < 0: det *= -1
        if det < MIN_IMPURITY_SPLIT: det = MIN_IMPURITY_SPLIT
        return np.log(det)
 
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
         
        
    # !TODO: This function does not differ from the parent version. Why do I get error, when I remove it?
    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
 
        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features
 
        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state
 
        cdef SplitRecord best, current
 
        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
 
        _init_split(&best, end)
 
        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
 
            n_visited_features += 1
 
            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.
 
            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)
 
            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp
 
                n_drawn_constants += 1
 
            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
 
                current.feature = features[f_j]
 
                # Sort samples along that feature; first copy the feature
                # values for the active samples into Xf, s.t.
                # Xf[i] == X[samples[i], j], so the sort uses the cache more
                # effectively.
                for p in range(start, end):
                    Xf[p] = X[X_sample_stride * samples[p] +
                              X_fx_stride * current.feature]
 
                sort(Xf + start, samples + start, end - start)
                #self.criterion_real.sortS()        
 
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature
 
                    n_found_constants += 1
                    n_total_constants += 1
 
                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]
 
                    # Evaluate all splits
                    self.criterion.reset()
                    p = start
 
                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1
 
                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])
 
                        if p < end:
                            current.pos = p
 
                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue
 
                            self.criterion.update(current.pos)
 
                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue
 
                            current.improvement = self.criterion.impurity_improvement(impurity)
 
                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0
 
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]
 
                                best = current  # copy
 
        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start
 
            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best.feature] <= best.threshold:
                    p += 1
 
                else:
                    partition_end -= 1
 
                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp
 
        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
 
        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)
 
        # Return values
        #with gil: print 'e-best', best.pos, best.feature, best.threshold
        split[0] = best
        n_constant_features[0] = n_total_constants

cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return low + our_rand_r(random_state) % (high - low)

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    
# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


#cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
#    # Helper for sort
#    Xf[i], Xf[j] = Xf[j], Xf[i]
#    samples[i], samples[j] = samples[j], samples[i]

cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)

cdef test():
    cdef DTYPE_t X[3]
    X[0] = 2.0
    X[1] = 1.0
    X[2] = 0.0
    cdef SIZE_t s[3]
    s[0] = 3
    s[1] = 2
    s[2] = 1
    cdef SIZE_t n = 3
    cdef int m = 5
    introsort(X, s, n, m)
    
test()
print X, s