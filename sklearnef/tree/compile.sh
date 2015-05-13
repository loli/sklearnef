#!/bin/bash

VERSION="0.16.1"

echo '---- CLEAN ----'
rm headers/${VERSION}/_tree.c
rm _tree.so

echo '---- COMPILE ----'
cython _tree.pyx -o headers/${VERSION}/_tree.c -I headers/ -f
python setup.py build_ext --inplace
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o _tree.so _tree.c

echo '---- RUN ----'
ipython -c "import _tree"

