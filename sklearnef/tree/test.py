#!/usr/bin/env python

from tree import UnSupervisedDecisionTreeClassifier
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.

clf = UnSupervisedDecisionTreeClassifier()
clf.fit(X)

