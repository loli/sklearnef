#!/bin/bash

VERSION="0.16.1"

echo '---- CLEAN ----'
rm headers/${VERSION}/_tree.c
rm _tree.so
rm headers/_diffentropy.c
rm _diffentropy.so

echo '---- COMPILE ----'
cython _diffentropy.pyx -o headers/_diffentropy.c -I headers/ -f
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o _diffentropy.so _diffentropy.c
cython _tree.pyx -o headers/${VERSION}/_tree.c -I headers/ -f
python setup.py build_ext --inplace

echo '---- RUN ----'
ipython -c "import _tree"

