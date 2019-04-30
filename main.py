#!/usr/bin/env python3

import pickle
import preprocessing
import feature_extraction
import train_svm
import numpy as np

# Check for existing model.
try:
    with open("X", "rb") as fd:
        X = pickle.load(fd)

    with open("y", "rb") as fd:
        y = pickle.load(fd)

    with open("X_test", "rb") as fd:
        X_test = pickle.load(fd)

    with open("y_test", "rb") as fd:
        y_test = pickle.load(fd)

    with open("X_val", "rb") as fd:
        X_val = pickle.load(fd)

    with open("y_val", "rb") as fd:
        y_val = pickle.load(fd)
# Build new model if no existing model found.
except OSError:
    import create_datasets
    # We use the words occurring in the training set at least as often as the
    # given treshold as features for the model.
    most_common_words_dict = preprocessing.preprocess_dir("trainingset/",
                                                          count_words=True,
                                                          treshold=3)
    preprocessing.preprocess_dir("valset/")
    preprocessing.preprocess_dir("testset/")

    most_common_words = list(most_common_words_dict)
    with open("most_common_words", "wb") as fd:
        pickle.dump(most_common_words, fd)

    print("Extracting features...")
    X, y = feature_extraction.extract_features_from_dir("trainingset/",
                                                        most_common_words)
    with open("X", "wb") as fd:
        pickle.dump(X, fd)

    with open("y", "wb") as fd:
        pickle.dump(y, fd)

    X_test, y_test = feature_extraction.extract_features_from_dir("testset/",
                                                                  most_common_words)
    with open("X_test", "wb") as fd:
        pickle.dump(X_test, fd)

    with open("y_test", "wb") as fd:
        pickle.dump(y_test, fd)

    X_val, y_val = feature_extraction.extract_features_from_dir("valset/",
                                                                most_common_words)
    with open("X_val", "wb") as fd:
        pickle.dump(X_val, fd)

    with open("y_val", "wb") as fd:
        pickle.dump(y_val, fd)

print("Training classifier...")

clf = train_svm.train_classifier(X, y, X_val, y_val, max_iter=1000)
with open("clf", "wb") as fd:
    pickle.dump(clf, fd)

print("\nSupport Vector classifier successfully trained.\n")
predictions = clf.predict(X_test)
ground_truth = np.ravel(y_test)

tp = np.sum(ground_truth == predictions)
fp = np.sum(ground_truth * np.logical_not(predictions))
fn = np.sum(np.logical_not(predictions) * predictions)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (np.sum(ground_truth == predictions) / len(ground_truth)) * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")
