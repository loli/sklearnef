"""
Testing for the _diffentropy module (sklearnef.tree._diffentropy).
"""

import functools
import nose
import numpy
from numpy.testing import assert_array_equal, assert_equal, assert_array_almost_equal, assert_almost_equal

from sklearnef.tree import _diffentropy

####
# Constants
####
INTYPE_t = numpy.float32
DTYPE_t = numpy.float64
SIZE_t = numpy.intp

####
# Decorator to convert caught AttributeErrors to SkipTest exceptions
####
def skip_on_attribute_error(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        try:
            fun(*args, **kwargs)
        except AttributeError:
            raise nose.SkipTest
    return wrapper

####
# Helper functions
####
def __cov_det_logdet_limes_numpy(X):
    "Compute numpy-version of logdetcov with correct border condition."
    singularity_threshold = _diffentropy._get_singularity_threshold()
    cov = numpy.cov(X, rowvar=0, ddof=1)
    det = numpy.linalg.det(cov)
    logdet = numpy.log(singularity_threshold) if det < singularity_threshold else numpy.log(det)
    return cov[numpy.triu_indices(X.shape[1])], det, logdet, det < singularity_threshold

def __cov_logdet_diffentropy(X):
    "Compute cython-version of logdetcov."
    obj = _diffentropy.Diffentropy(X.shape[1])
    for _x in X:
        _diffentropy._call_update_add(obj, _x)
    _diffentropy._call_compute_covariance_matrix(obj)
    return _diffentropy._get_cov(obj), _diffentropy._call_logdet(obj)

####
# Conditional test for C-functionality.
####
@skip_on_attribute_error
def test_toy_example_limes():
    X = numpy.asarray([[4.0, 2.0, 0.60],
                       [4.2, 2.1, 0.59],
                       [3.9, 2.0, 0.58],
                       [4.3, 2.1, 0.62],
                       [4.1, 2.2, 0.63]]).astype(INTYPE_t)
    ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X)
    ccov, clogdet = __cov_logdet_diffentropy(X)
    assert_array_almost_equal(ncov, ccov)
    assert_almost_equal(nlogdet, clogdet)

def test_toy_example_nonlimes():
    numpy.random.seed(0)
    X = numpy.random.rand(100, 2).astype(INTYPE_t)
    ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X)
    ccov, clogdet = __cov_logdet_diffentropy(X)
    assert_array_almost_equal(ncov, ccov)
    assert_almost_equal(nlogdet, clogdet)
    
def test_rank_deficiency():
    numpy.random.seed(0)
    X = numpy.random.rand(3, 20).astype(INTYPE_t)
    ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X)
    ccov, clogdet = __cov_logdet_diffentropy(X)
    assert_array_almost_equal(ncov, ccov)
    assert_almost_equal(nlogdet, clogdet)

def test_shifted_data():
    numpy.random.seed(0)
    X = numpy.random.rand(100, 2).astype(INTYPE_t)
    ccov1, clogdet1 = __cov_logdet_diffentropy(X)
    ccov2, clogdet2 = __cov_logdet_diffentropy(X + 10)
    assert_array_almost_equal(ccov1, ccov2)
    assert_almost_equal(clogdet1, clogdet2, decimal=5)

def test_add_samples_incrementally():
    numpy.random.seed(0)
    X = numpy.random.rand(100, 2).astype(INTYPE_t)
    obj = _diffentropy.Diffentropy(X.shape[1])
    
    for i, _x in enumerate(X):
        _diffentropy._call_update_add(obj, _x)
        if i >= 1:
            _diffentropy._call_compute_covariance_matrix(obj)
            ccov, clogdet = _diffentropy._get_cov(obj), _diffentropy._call_logdet(obj)
            ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X[:i+1])
            assert_array_almost_equal(ncov, ccov)
            assert_almost_equal(nlogdet, clogdet)
            
def test_sub_samples_incrementally():
    numpy.random.seed(0)
    X = numpy.random.rand(100, 2).astype(INTYPE_t)
    obj = _diffentropy.Diffentropy(X.shape[1])
    
    for _x in X:
        _diffentropy._call_update_add(obj, _x)
    
    for i, _x in enumerate(X):
        if i < len(X) - 1:
            _diffentropy._call_compute_covariance_matrix(obj)
            ccov, clogdet = _diffentropy._get_cov(obj), _diffentropy._call_logdet(obj)
            ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X[i:])
            assert_array_almost_equal(ncov, ccov)
            assert_almost_equal(nlogdet, clogdet)
        
        _diffentropy._call_update_sub(obj, _x)
    
def test_reset():
    numpy.random.seed(0)
    X = numpy.random.rand(100, 2).astype(INTYPE_t)
    obj = _diffentropy.Diffentropy(X.shape[1])
    
    # default use all of the samples
    for _x in X:
        _diffentropy._call_update_add(obj, _x)
    _diffentropy._call_compute_covariance_matrix(obj)
    ccov, clogdet = _diffentropy._get_cov(obj), _diffentropy._call_logdet(obj)
    ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X)
    assert_array_almost_equal(ncov, ccov)
    assert_almost_equal(nlogdet, clogdet)
    
    # reset
    _diffentropy._call_reset(obj)
    
    # after reset use only half of the samples
    for _x in X[:-X.shape[0]/2]:
        _diffentropy._call_update_add(obj, _x)
    _diffentropy._call_compute_covariance_matrix(obj)
    ccov, clogdet = _diffentropy._get_cov(obj), _diffentropy._call_logdet(obj)
    ncov, ndet, nlogdet, nlimes = __cov_det_logdet_limes_numpy(X[:-X.shape[0]/2])
    assert_array_almost_equal(ncov, ccov)
    assert_almost_equal(nlogdet, clogdet)

####
# Conditional tests for cdef functions.
# If test wrappers not available from module _diffentropy, we assume they have been removed to reduce file size and skip the tests.
####


@skip_on_attribute_error
def test_vector_multiply_scalar():
    length = 10
    x = numpy.random.random(length).astype(DTYPE_t)
    a = numpy.random.random()
    
    expected_result = x * a
    _diffentropy._test_wrapper_vector_multiply_scalar(x, a, length)
    assert_array_equal(x, expected_result)

@skip_on_attribute_error
def test_upper_sub_outer_product_eachway():
    length = 3
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    y = x[::-1].copy()
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(i, length + 1):
            expected_result[c] = -1 * (i * (length - j + 1) + j * (length - i + 1))
            c += 1      
    
    # run and test
    _diffentropy._test_wrapper_upper_sub_outer_product_eachway(X, x, y, length)
    assert_array_equal(X, expected_result)

@skip_on_attribute_error
def test_vector_add():
    length = 10
    x = numpy.asarray(range(length), dtype=DTYPE_t)
    y = x[::-1].copy()
    expected_result = numpy.asarray([length-1] * length, dtype=DTYPE_t)
    
    _diffentropy._test_wrapper_vector_add(x, y, length)
    assert_array_equal(x, expected_result)
    
@skip_on_attribute_error
def test_vector_sub():
    length = 10
    x = numpy.asarray(range(length), dtype=DTYPE_t)
    y = x[::-1].copy()
    expected_result = numpy.asarray(range(-1 * length + 1, length, 2), dtype=DTYPE_t)
    
    _diffentropy._test_wrapper_vector_sub(x, y, length)
    assert_array_equal(x, expected_result)    
    
@skip_on_attribute_error
def test_upper_add_sample_autocorrelation_matrix():
    length = 5
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(0, length - i + 1):
            expected_result[c] = (j + i) * i
            c += 1    
    
    # run and test
    _diffentropy._test_wrapper_upper_add_sample_autocorrelation_matrix(X, x, length)
    assert_array_equal(X, expected_result)    
    
@skip_on_attribute_error
def test_upper_sub_sample_autocorrelation_matrix():
    length = 5
    x = numpy.asarray(range(length), dtype=DTYPE_t) + 1
    X = numpy.zeros((length * length + length)/2, dtype=DTYPE_t)
    expected_result = numpy.zeros(X.shape[0], dtype=DTYPE_t)
    
    # compute expected results
    c = 0
    for i in range(1, length + 1):
        for j in range(0, length - i + 1):
            expected_result[c] = -1 * (j + i) * i
            c += 1    
    
    # run and test
    _diffentropy._test_wrapper_upper_sub_sample_autocorrelation_matrix(X, x, length)
    assert_array_equal(X, expected_result)
    
@skip_on_attribute_error
def test_upper_n_elements():
    length = 99
    expected_result = 0
        
    for i in range(1, length + 1):
        expected_result += i
        
    result = _diffentropy._test_wrapper_n_elements(length)
    assert_equal(result, expected_result)
    