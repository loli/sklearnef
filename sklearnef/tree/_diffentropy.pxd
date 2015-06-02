# python imports

# cython imports
cimport numpy as np

# extern cdefs: lapack c-wrapped fortran routine definitions
# A = U*D*U**T // http://www.netlib.org/lapack/explore-html/d1/dcd/dsptrf_8f.html
cdef extern void dsptrf_( char *uplo, int *n, double *ap, int *ipiv, int *info ) nogil

# type definitions
ctypedef np.npy_float32 INTYPE_t # input data type, should be of leass or equl precision that internal data type
ctypedef np.npy_float64 DTYPE_t  # data type for internal calculations
ctypedef np.npy_intp SIZE_t      # type for indices and counters

cdef class Diffentropy:
    # Dynamically updateable statistical co-variance matrix and log-determinant computation with delayed update
    # based on: https://github.com/loli/dynstatcov
    #           and
    #           https://github.com/loli/logdet
    
    # internal structure
    cdef DTYPE_t* cov           # upper triangular part of the co-variance matrix
    cdef DTYPE_t* squaresum     # upper triangular part of the sum of all samples outer product
    cdef DTYPE_t* sum           # sum of all samples
    cdef SIZE_t n_samples       # number of samples from which the co-variance matrix is computed
    cdef SIZE_t n_features      # number of elements per samples
    cdef SIZE_t n_upper         # elements in the upper triangular matrix
    
    cdef DTYPE_t* __mean        # private member
    cdef int* __ipiv            # private member
    cdef double* __covf         # private member
    
    cdef void update_add(self, INTYPE_t* x) nogil       # no-gil update method (addition)
    cdef void update_sub(self, INTYPE_t* x) nogil       # no-gil update method (subtraction)
    cdef void reset(self) nogil                        # reset the class
    cdef void compute_covariance_matrix(self) nogil    # trigger the computation of the co-variance matrix
    cdef DTYPE_t logdet(self) nogil                       # return the log-determinant of the co-variance matrix