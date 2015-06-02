# encoding: utf-8
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

"""
A run-time optimized library for dynamic updates of a statistical co-variance matrix
and differential entropy (i.e. log-determinant) computation.

This class is a specialization of the methods in https://github.com/loli/dynstatcov
and https://github.com/loli/logdet to make use of a common memory space as well as
a delayed co-variance matrix update.
"""

##########
# Changelog
# 2015-06-01 delayed-updated and logdet added
# 2015-02-22 added a subtraction option for update()
# 2015-02-20 properly documented and tested
# 2015-02-17 created
##########

# python imports

# cython imports
cimport numpy as np
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log, sqrt

# extern cdefs

# constants
cdef double SINGULARITY_THRESHOLD = 1e-6

# type definitions

# docstring info
__author__ = "Oskar Maier"
__copyright__ = "Copyright 2015, Oskar Maier"
__version__ = "0.1.1"
__maintainer__ = "Oskar Maier"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"

cdef class Diffentropy:
    r"""
    Dynamically updateable statistical co-variance matrix for logdet computation.
    
    Implements a minimal method for dynamic updates of a statistic co-variance
    matrix upon the arrival of a new observation. Optimized for speed and low
    memory requirements: the computational and memory requirements depend
    only on the number of features in each observation vector and not on the
    number of observations.
    
    A lapack-based computation of the co-variance matrixes log-det is also
    possible.    
    
    Parameters
    ----------
    n_features : int
        The number of features (i.e. co-variance matrix sidelength).
           
    Notes
    -----
    Diffentropy can be compiled with either single (`numpy.float32`) or
    double precision (`numpy.float64`). You might have to test, which version
    you are running. By default, its double precision.
    If required, you can always change typedef of `DTYPE_t` in the cython code
    and re-compile.
    """

    def __cinit__(self, SIZE_t n_features):
        cdef SIZE_t n_samples = 0
        cdef SIZE_t n_upper = upper_n_elements(n_features)
         
        #cdef DTYPE_t* squaresum = <DTYPE_t*> PyMem_Malloc(n_upper * sizeof(DTYPE_t))
        #cdef DTYPE_t* sum = <DTYPE_t*> PyMem_Malloc(n_features * sizeof(DTYPE_t))
          
        cdef SIZE_t i = 0
          
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_upper = n_upper
        self.squaresum = <DTYPE_t*> calloc(n_upper, sizeof(DTYPE_t))
        self.sum = <DTYPE_t*> calloc(n_features, sizeof(DTYPE_t))
        #self.cov = <DTYPE_t*> PyMem_Malloc(n_upper * sizeof(DTYPE_t))
        self.cov = <DTYPE_t*> malloc(n_upper * sizeof(DTYPE_t))
        #self.__mean = <DTYPE_t*> PyMem_Malloc(n_features * sizeof(DTYPE_t))
        self.__mean = <DTYPE_t*> malloc(n_features * sizeof(DTYPE_t))
        self.__ipiv = <int*> malloc(n_features * sizeof(int))
        self.__covf = <double*> malloc(n_upper * sizeof(double))
          
        if not self.cov or not self.squaresum or not self.sum or \
           not self.__mean or not self.__ipiv or not self.__covf:
            raise MemoryError()
        
        # initialize with 0s
        #fill_zeros(sum, n_features)
        #fill_zeros(squaresum, n_upper)
        
        # initialize cov matrix construction elements        
        #for i in range(n_samples):
        #    vector_add(sum, &X[i][0], n_features) # or X.buf + i * n_features ?
        #    upper_add_sample_autocorrelation_matrix(squaresum, &X[i][0], n_features)
              
        #self.__compute_covariance_matrix()
        
    def __dealloc__(self):
        free(self.cov)
        free(self.squaresum)
        free(self.sum)
        free(self.__mean)
        free(self.__ipiv)
        free(self.__covf)
        
    cdef void update_add(self, INTYPE_t* x) nogil:
        "Add a new sample without updating the co-variance matrix."
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        
        n_samples += 1
        vector_add_in(sum, x, n_features)
        upper_add_sample_autocorrelation_matrix_in(squaresum, x, n_features)
        
        self.n_samples = n_samples
        
    cdef void update_sub(self, INTYPE_t* x) nogil:
        "Remove a sample without updating the co-variance matrix."
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        
        n_samples -= 1
        vector_sub_in(sum, x, n_features)
        upper_sub_sample_autocorrelation_matrix_in(squaresum, x, n_features)
        
        self.n_samples = n_samples
        
    cdef void reset(self) nogil:
        "Reset all sample and the co-variance matrix."
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_upper = self.n_upper
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        
        # reset counters        
        self.n_samples = 0
        
        # fill with 0s
        fill_zeros(sum, n_features)
        fill_zeros(squaresum, n_upper)
        
    cdef void compute_covariance_matrix(self) nogil:
        "Compute the co-variance matrix from its components."
        cdef DTYPE_t* cov = self.cov
        cdef DTYPE_t* sum = self.sum
        cdef DTYPE_t* squaresum = self.squaresum
        cdef SIZE_t n_samples = self.n_samples
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_upper = self.n_upper
        
        cdef DTYPE_t* mean = self.__mean
        
        # create mean of samples sum vector
        fill_zeros(mean, n_features)
        vector_add(mean, sum, n_features)
        vector_multiply_scalar(mean, 1.0/n_samples, n_features)
        
        # compute co-variance matrix
        fill_zeros(cov, n_upper)
        upper_add_sample_autocorrelation_matrix(cov, mean, n_features)
        vector_multiply_scalar(cov, n_samples, n_upper)
        upper_sub_outer_product_eachway(cov, mean, sum, n_features)
        vector_add(cov, squaresum, n_upper)
        vector_multiply_scalar(cov, 1.0/(n_samples - 1), n_upper)
        
    cdef DTYPE_t logdet(self) nogil:
        "Compute the log-determinant of the co-variance matrix."
        cdef DTYPE_t* cov = self.cov
        cdef double* covf = self.__covf
        cdef int* ipiv = self.__ipiv
        cdef SIZE_t n_features = self.n_features
        
        cdef int i, info
        cdef double logdet, tmp
        
        # to fortran order (column-major), copies data
        asfortran(covf, cov, n_features)
        
        # solves A = U*D*U**T => output A = D
        dsptrf_('U', <int*>&n_features, covf, ipiv, &info)
    
        # Interpretation of info:
        # info == 0: all is fine
        # info > 0:  system is singular (D has a 0 on the diagonal), mutliplication with 0 will occur for det, det will be 0
        # info < 0:  invalid argument at position abs(info)
        
        # assemble logdet
        logdet = 0.
        for i in range(n_features):
            if ipiv[i] > 0:
                tmp = covf[ umat(i,i) ]
                if tmp < SINGULARITY_THRESHOLD: return log(SINGULARITY_THRESHOLD)
                logdet += log(tmp)
            elif i > 0 and ipiv[i] < 0 and ipiv[i-1] == ipiv[i]:
                tmp = covf[ umat(i,i) ] * covf[ umat(i-1,i-1) ] -\
                       covf[ umat(i-1,i) ] * covf[ umat(i-1,i) ]
                if tmp < SINGULARITY_THRESHOLD: return log(SINGULARITY_THRESHOLD)
                logdet += log(tmp)

        return <DTYPE_t>logdet if logdet > log(SINGULARITY_THRESHOLD) else <DTYPE_t>log(SINGULARITY_THRESHOLD)
        
cdef inline void asfortran(double* F, DTYPE_t* C, SIZE_t n) nogil:
    """
    Takes a condensed form upper triangular matrix in
    C order (row-major) and converts it to fortran order
    (column-major) in condensed format, saving it into
    the passed memory.
    """
    cdef:
        SIZE_t i, j
    
    for j in range(n):
        for i in range(n - j):
            F[idx_f(i, j)] = <double>C[idx_c(i, j, n)]

cdef inline SIZE_t idx_f(SIZE_t i, SIZE_t j) nogil:
    """
    Compute the fortran-style index of a condensed upper triangular matrix.
    Assuming (slower) j<-0:n and (faster) i<-0:n-j.
    """
    return j + (i + j) * (i + j + 1) / 2

cdef inline SIZE_t idx_c(SIZE_t i, SIZE_t j, SIZE_t n) nogil:
    """
    Compute the C-style index of a condensed upper triangular matrix.
    Assuming (slower) j<-0:n and (faster) i<-0:n-j.
    """
    return j * n - j * (j + 1) / 2 + i + j

cdef inline int umat(int i, int j) nogil:
    return i + j * ( j + 1 ) / 2        
        
cdef inline void vector_multiply_scalar(DTYPE_t* X, DTYPE_t a, SIZE_t length) nogil:
    "Multiply all elements of the matrix X with a."
    cdef SIZE_t p = 0
    
    for p in range(length):
        X[p] *= a
        
cdef inline void upper_sub_outer_product_eachway(DTYPE_t* X, DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Substract the outer product of x and y as well as y and x from the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] -= x[p1] * y[p2] + x[p2] * y[p1]
            X += 1

cdef inline void fill_zeros(DTYPE_t* x, SIZE_t length) nogil:
    "Fill an array with zeros"
    cdef SIZE_t i = 0
    
    for i in range(length):
        x[i] = 0.0

cdef inline SIZE_t upper_n_elements(SIZE_t n) nogil:
    "The number of (diagonal including) elements of an upper triangular nxn matrix."
    return (n * n + n) / 2

cdef inline void vector_add(DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Add vectors y to vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] += y[p]
        
cdef inline void vector_add_in(DTYPE_t* x, INTYPE_t* y, SIZE_t length) nogil:
    "Add vectors y to vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] += <DTYPE_t>y[p]
        
cdef inline void vector_sub(DTYPE_t* x, DTYPE_t* y, SIZE_t length) nogil:
    "Subtract vector y from vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] -= y[p]    
        
cdef inline void vector_sub_in(DTYPE_t* x, INTYPE_t* y, SIZE_t length) nogil:
    "Subtract vector y from vector x."
    cdef SIZE_t p = 0
    
    for p in range(length):
        x[p] -= <DTYPE_t>y[p]           

cdef inline void upper_add_sample_autocorrelation_matrix(DTYPE_t* X, DTYPE_t* x, SIZE_t length) nogil:
    "Add the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] += x[p1] * x[p2]
            X += 1
            
cdef inline void upper_add_sample_autocorrelation_matrix_in(DTYPE_t* X, INTYPE_t* x, SIZE_t length) nogil:
    "Add the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] += <DTYPE_t>x[p1] * <DTYPE_t>x[p2]
            X += 1            
            
cdef inline void upper_sub_sample_autocorrelation_matrix(DTYPE_t* X, DTYPE_t* x, SIZE_t length) nogil:
    "Subtract the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] -= x[p1] * x[p2]
            X += 1
            
cdef inline void upper_sub_sample_autocorrelation_matrix_in(DTYPE_t* X, INTYPE_t* x, SIZE_t length) nogil:
    "Subtract the outer product of x with itself to the upper triangular part of X."
    cdef SIZE_t p1 = 0
    cdef SIZE_t p2 = 0
    
    for p1 in range(length):
        for p2 in range(p1, length):
            X[0] -= <DTYPE_t>x[p1] * <DTYPE_t>x[p2]
            X += 1            
     
#####
# Wrapper to call C-functions of the class for unittesting. Can be deleted if no tests are used.
#####
cpdef _call_update_add(Diffentropy obj, INTYPE_t[::1] x):
    obj.update_add(&x[0])
    
cpdef _call_update_sub(Diffentropy obj, INTYPE_t[::1] x):
    obj.update_sub(&x[0])
    
cpdef _call_reset(Diffentropy obj):
    obj.reset()
    
cpdef _call_compute_covariance_matrix(Diffentropy obj):
    obj.compute_covariance_matrix()
    
cpdef double _call_logdet(Diffentropy obj):
    return obj.logdet()

cpdef np.ndarray _get_cov(Diffentropy obj):
    import numpy
    
    cdef SIZE_t n_upper = obj.n_upper
    cdef DTYPE_t* cov = obj.cov
    
    cdef DTYPE_t [::1] cov_view = <DTYPE_t[:n_upper]> cov
    return numpy.asarray(cov_view).copy()
            
cpdef double _get_singularity_threshold():
    return SINGULARITY_THRESHOLD
            
#####
# Cpdef functions to expose cdef function for unittesting. Can be deleted if no tests are used.
#####
cpdef int _test_wrapper_n_elements(SIZE_t length):
    return upper_n_elements(length)

cpdef _test_wrapper_upper_add_sample_autocorrelation_matrix(DTYPE_t[::1] X, DTYPE_t[::1] x, SIZE_t length):
    upper_add_sample_autocorrelation_matrix(&X[0], &x[0], length)
    
cpdef _test_wrapper_upper_sub_sample_autocorrelation_matrix(DTYPE_t[::1] X, DTYPE_t[::1] x, SIZE_t length):
    upper_sub_sample_autocorrelation_matrix(&X[0], &x[0], length)    
        
cpdef _test_wrapper_vector_add(DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    vector_add(&x[0], &y[0], length)
    
cpdef _test_wrapper_vector_sub(DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    vector_sub(&x[0], &y[0], length)    
    
cpdef _test_wrapper_vector_multiply_scalar(DTYPE_t[::1] X, DTYPE_t a, SIZE_t length):
    vector_multiply_scalar(&X[0], a, length)

cpdef _test_wrapper_upper_sub_outer_product_eachway(DTYPE_t[::1] X, DTYPE_t[::1] x, DTYPE_t[::1] y, SIZE_t length):
    upper_sub_outer_product_eachway(&X[0], &x[0], &y[0], length)

