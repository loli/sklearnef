# Authors: Oskar Maier <oskar.maier@googlemail.com>
#
# Licence: BSD 3 clause !TODO: Change!

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

#cdef extern from *:
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from sklearn.tree._tree cimport Criterion
#from sklearn_tree cimport Criterion

# =============================================================================
# Criterion
# =============================================================================

cdef class UnSupervisedClassificationCriterion(Criterion):
    # Internal structures
    cdef DTYPE_t* X             # pointer to the training data; will become obsolute, when I've implemented sort(S) without re-copying the memory
    cdef SIZE_t X_stride        # the X_stride; is this the same as n_features... might just be, then I can remove the second and use this in __reduce__
    cdef DTYPE_t* S             # copy of the training data for fitting multi-variate Gaussians
    cdef SIZE_t n_samples       # might not be required, as only used in __cinit__ and __reduce__
    cdef SIZE_t n_features      # might not be required, as only used in __cinit__ and __reduce__
    # !TODO: Above, some of the member vars could be removed in the future.
     
    # Methods
    cdef void init2(self, DTYPE_t* X, SIZE_t X_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil
    #!TODO: sortS should actually not be a method, but be performed in impurity_improvement together with sort(xf, samples, ...)
    cdef void sortS(self) nogil
    cdef double differential_entropy(self, DTYPE_t* src, SIZE_t size)