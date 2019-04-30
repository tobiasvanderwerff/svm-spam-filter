#!/usr/bin/env python3

import os
import sys
import random
import shutil
from math import floor

print("Creating data sets...")

# There should be a directory 'spam' and 'ham' containing plain text
# emails: one containing spam emails and the other containing non-spam
# emails, respectively.
try:
    spam = os.listdir("spam")
    ham = os.listdir("ham")
except FileNotFoundError:
    print("'ham' and/or 'spam' directory not found, exiting.")
    sys.exit()

total_spam = len(spam)
total_ham = len(ham)

random.shuffle(ham)
random.shuffle(spam)

# Distribution of data: training set 60%, cross-validation set 20%, test
# set 20%.
train_ham = ham[:floor(total_ham * (6 / 10))]
val_ham = ham[floor(total_ham * (6 / 10)):floor(total_ham * (8 / 10))]
test_ham = ham[floor(total_ham * (8 / 10)):]

train_spam = spam[:floor(total_spam * (6 / 10))]
val_spam = spam[floor(total_spam * (6 / 10)):floor(total_spam * (8 / 10))]
test_spam = spam[floor(total_spam * (8 / 10)):]

try:
    os.mkdir("trainingset")
except FileExistsError:
    # Note that if the directory already exists, we simply delete it.
    shutil.rmtree("trainingset")
    os.mkdir("trainingset")

try:
    os.mkdir("valset")
except FileExistsError:
    shutil.rmtree("valset")
    os.mkdir("valset")

try:
    os.mkdir("testset")
except FileExistsError:
    shutil.rmtree("testset")
    os.mkdir("testset")

for i, f in enumerate(train_ham):
    shutil.copyfile(f"ham/{f}", f"trainingset/ham_train{i}")

for i, f in enumerate(train_spam):
    shutil.copyfile(f"spam/{f}", f"trainingset/spam_train{i}")

for i, f in enumerate(val_ham):
    shutil.copyfile(f"ham/{f}", f"valset/ham_val{i}")

for i, f in enumerate(val_spam):
    shutil.copyfile(f"spam/{f}", f"valset/spam_val{i}")

for i, f in enumerate(test_ham):
    shutil.copyfile(f"ham/{f}", f"testset/ham_test{i}")

for i, f in enumerate(test_spam):
    shutil.copyfile(f"spam/{f}", f"testset/spam_test{i}")
