#!/usr/bin/env python3

import pickle
import preprocessing
import feature_extraction
import train_svm

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
    most_common_words_dict = preprocessing.preprocess_data("trainingset/",
                                                           count_words=True,
                                                           treshold=100)
    preprocessing.preprocess_data("valset/")
    preprocessing.preprocess_data("testset/")

    most_common_words = most_common_words_dict.keys()

    print("Extracting features...")
    X, y = feature_extraction.extract_features("trainingset/",
                                               most_common_words)
    with open("X", "wb") as fd:
        pickle.dump(X, fd)

    with open("y", "wb") as fd:
        pickle.dump(y, fd)

    X_test, y_test = feature_extraction.extract_features("testset/",
                                                         most_common_words)
    with open("X_test", "wb") as fd:
        pickle.dump(X_test, fd)

    with open("y_test", "wb") as fd:
        pickle.dump(y_test, fd)

    X_val, y_val = feature_extraction.extract_features("valset/",
                                                       most_common_words)
    with open("X_val", "wb") as fd:
        pickle.dump(X_val, fd)

    with open("y_val", "wb") as fd:
        pickle.dump(y_val, fd)

print("Training classifier...")
clf = train_svm.train_classifier(X, y, X_val, y_val, max_iter=1000)
print("Support Vector classifier successfully trained.")
accuracy = clf.score(X_test, y_test)
print(f"Spam classification accuracy: {accuracy * 100:.2f}%")