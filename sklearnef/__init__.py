"""
Extension for the scikit-learn machine learning module for Python
=================================================================

sklearnef
---------
sklearnef extends the random decision forest functionality of sklearn
by un- and semi-supervised random decision forest implementations.

https://pypi.python.org/pypi/sklearnef/
https://github.com/loli/sklearnef/


sklearn
-------
sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.org for complete documentation.


Copyright (C) 2013 Oskar Maier, <oskar.maier@googlemail.com>

!TODO: Add a license.
"""
import sys
import re
import warnings

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
#
__version__ = '0.1.dev'
__all__ = ['ensemble', 'tree']
