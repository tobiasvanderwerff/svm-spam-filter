#!/usr/bin/env python3

import numpy as np
import collections
import os


def extract_features(data_dir, most_common_words):
    n = len(most_common_words)                  # No. of features
    m = len(os.listdir(data_dir))               # No. of training examples
    X = np.zeros((m, n))
    y = np.zeros((m, 1))
    for i, f in enumerate(os.listdir(data_dir)):
        with open(data_dir + f, "r") as fd:
            data = fd.read()
            fd_words = data.split()
            fd_word_count = collections.Counter(fd_words)
            # print(f"Word count for preprocessed_mail/{f}:\n {fd_word_count}")
            for j, word in enumerate(most_common_words):
                if word in fd_word_count:
                    X[i, j] = fd_word_count[word]
                    # print(f"X[{i}, {j}] = {fd_word_count[word]}")
            if f[:4] == "spam":
                y[i] = 1
    return X, y
