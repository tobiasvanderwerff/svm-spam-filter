#!/usr/bin/env python3

import preprocessing
import pickle
import sys
import argparse
from feature_extraction import extract_features_from_word_list

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="file containing the source of an email")
args = parser.parse_args()
filename = args.filename

words = preprocessing.preprocess_file(filename)
if words is None:
    print("Please provide a utf-8 formatted email.")
else:
    try:
        with open("most_common_words", "rb") as fd:
            most_common_words = pickle.load(fd)
    except OSError:
        print("most_common_words not found; exiting.")
        sys.exit()
    try:
        with open("clf", "rb") as fd:
            clf = pickle.load(fd)
    except OSError:
        print("Existing classifier not found; exiting.")
        sys.exit()
    x = extract_features_from_word_list(words, most_common_words)
    res = clf.predict(x)[0]
    if res == 1:
        print("Spam")
    else:
        print("Ham")
