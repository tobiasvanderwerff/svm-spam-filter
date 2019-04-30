#!/usr/bin/env python3

import numpy as np
import collections
import os


def extract_features_from_dir(dir_name, most_common_words):
    n = len(most_common_words)                  # No. of features
    m = len(os.listdir(dir_name))               # No. of training examples
    X = np.zeros((m, n))
    y = np.zeros((m, 1))
    for i, f in enumerate(os.listdir(dir_name)):
        with open(dir_name + f, "r") as fd:
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


def extract_features_from_word_list(word_list, most_common_words):
    x = np.zeros((len(most_common_words), 1))
    word_count = collections.Counter(word_list)
    for i, word in enumerate(most_common_words):
        if word in word_count:
            x[i] = word_count[word]
    return np.transpose(x)
