# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change!

# See _tree.pyx for details.

import numpy as np
cimport numpy as np
cimport sklearn.tree._tree
import sklearn.tree._tree

#cdef extern from *:
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# Import external extension types (with associated PXD files)
from sklearn.tree._tree cimport Criterion, Splitter, SplitRecord
from sklearnef.tree._diffentropy cimport Diffentropy

# Define external extension types (without associated PXD files)
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
# Criterion
# =============================================================================

cdef class UnSupervisedClassificationCriterion(Criterion):
    # Internal structures
    cdef DTYPE_t* X             # pointer to the training data; will become obsolute, when I've implemented sort(S) without re-copying the memory
    cdef SIZE_t X_stride        # the X_stride; is this the same as n_features... might just be, then I can remove the second and use this in __reduce__
    cdef SIZE_t n_samples       # might not be required, as only used in __cinit__ and __reduce__
    cdef SIZE_t n_features      # might not be required, as only used in __cinit__ and __reduce__
    cdef DTYPE_t min_improvement# minimal improvement of a split to consider it
    cdef DTYPE_t normalization_entropy # the node's initial entropy / logdet stored for normalization purposes
    cdef Diffentropy covr       # dynamically updateable covariance matrix (right)
    cdef Diffentropy covl       # dynamically updateable covariance matrix (left)
    # !TODO: Above, some of the member vars could be removed in the future.
    # The number of 'effective' prior observations (default = 0).
    #cdef DTYPE_t effprior
    # The variance of the effective observations (default = 900).
    #cdef DTYPE_t effpriorvar  
     
    # Methods
    cdef void init2(self, DTYPE_t* X, SIZE_t X_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil

cdef class SemiSupervisedClassificationCriterion(UnSupervisedClassificationCriterion):
    # Internal structures
    cdef DTYPE_t supervised_weight                                  # balancing weight
    cdef ClassificationCriterion criterion_supervised                    # supervised split quality criterion

    # Methods
    cdef void init3(self, DTYPE_t* X, SIZE_t X_stride, DOUBLE_t* y, SIZE_t y_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil
