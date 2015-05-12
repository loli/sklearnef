import os

import numpy
import sklearn
from numpy.distutils.misc_util import Configuration

def configuration(parent_package="", top_path=None):
    config = Configuration("tree", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
        
    # check for pre-compiled versions for the encountered sklearn version
    if not os.path.isdir("headers/{}".format(sklearn.__version__)) or \
       not os.path.isfile("headers/{}/_tree.c".format(sklearn.__version__)):
        raise Exception(\
"""sklearnef holds no pre-compiled _tree.c for your current scikit-learn version ({version}).
Please download the corresponding header file from \
https://raw.githubusercontent.com/scikit-learn/scikit-learn/{version}/sklearn/tree/_tree.pxd,
place it in sklearnef/tree/headers/{version}/_tree.pxd and compile _tree.pyx with cython using \
'cython _tree.pyx -o headers/{version}/_tree.c -I headers/{version}/_tree.pxd'. Then re-run \
the installation of sklearnef.""".format(version=sklearn.__version__))

    config.add_extension("_tree",
                         sources=["headers/{version}/_tree.c".format(version=sklearn.__version__)],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3"])
    
    config.add_subpackage("tests")

    return config

# def __fetch_sklearn_tree_header_file():
#     SKLEARN_REP_TREE_HEADER = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/{version}/sklearn/tree/_tree.pxd"
#     tmpdir = tempfile.mktemp(prefix="sklearnef")
#     tree_header = "{dir}_tree_{version}.pxd".format(dir=tmpdir,
#                                                     version=sklearn.__version__)
#     urllib.URLopener().retrieve(SKLEARN_REP_TREE_HEADER.format(sklearn.__version__),
#                                 tree_header)
#     
#     if not os.path.isfile(tree_header):
#         raise Exception("Failed to download the required sklearn headers for your \
#                          sklearn version ({version} from {url}.".format(
#                             version=sklearn.__version__,
#                             url=SKLEARN_REP_TREE_HEADER.format(sklearn.__version__)))
#     return tmpdir
# 
# def __clean_tmpdir(tmpdir):
#     tree_header = "{dir}_tree_{version}.pxd".format(dir=tmpdir,
#                                                     version=sklearn.__version__)
#     os.unlink(tree_header)
#     try:
#         os.rmdir(tmpdir)
#     except OSError as e:
#         warnings.warn("Failed to remove temporary directory '{dir}'. \
#                        Reason: {err}.".format(dir=tmpdir, err=e))

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())