#!/usr/bin/env python3

import numpy as np
import sklearn


def build_svc(X, y, C, max_iter):
    clf = sklearn.svm.LinearSVC(C=C, max_iter=max_iter)
    clf.fit(X, np.ravel(y))
    return clf


def train_classifier(X, y, X_val, y_val, max_iter=3000):
    C_vals = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    best_score = 0
    best_C = 0
    for C in C_vals:
        clf = build_svc(X, y, C, max_iter)
        crossval_score = clf.score(X_val, y_val)
        if crossval_score > best_score:
            best_score = crossval_score
            best_C = C
    clf = build_svc(X, y, best_C, max_iter)
    return clf
