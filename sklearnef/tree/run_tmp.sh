#!/bin/bash

LOC_HEADER="/home/maier/Workspacepython/sklearnef/scikit-learn-0.16.1/"
LOC_LIB="/home/maier/Workspacepython/sklearnef/venv/local/lib/python2.7/site-packages/sklearn/" # sklearn.__path__

echo '---- CLEAN ----'
rm _tree.c
rm _tree.so

echo '---- COMPILE ----'
cython _tree.pyx -I${LOC_HEADER}
python setup.py build_ext --inplace
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o _tree.so _tree.c

echo '---- RUN ----'
ipython -c "import _tree"

