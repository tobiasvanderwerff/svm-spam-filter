#!/usr/bin/env python3

import re
import nltk
import contractions
import os
import nltk.stem.snowball
import email.parser
import collections
from bs4 import BeautifulSoup


def parse_email(data):
    data = re.sub(r"http[s]?://[^\s]*", " httpaddr ", data)
    data = BeautifulSoup(data, "html.parser").text
    data = re.sub(r"[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[^\s]+", " emailaddr ", data)
    data = re.sub(r"[0-9]+(\.[0-9]+)*", " number ", data)
    data = re.sub(r"(\$|€)+", " currencyunit ", data)
    data = contractions.fix(data)   # replace contractions with expansions
    data = re.sub(r"[^a-zA-Z\s]+", "", data)
    data = data.lower()
    data = nltk.word_tokenize(data)

    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    data = [stemmer.stem(word) for word in data]
    return data


def preprocess_dir(dir_name, count_words=False, treshold=100):
    print(f"Preprocessing emails in {dir_name}...")
    if count_words:
        word_count = collections.Counter()
    for f in os.listdir(dir_name):
        with open(dir_name + f, "r") as rfd:
            try:
                data = email.parser.Parser().parse(rfd)
                body = data
                while body.is_multipart():
                    # MIME standard tells us the plain/text data of the email
                    # is placed at the top of the multipart message.
                    body = body.get_payload()[0]
                body = body.get_payload()
                words = parse_email(body)
                with open(f"{dir_name}{f}_parsed", "w") as wfd:
                    for word in words:
                        wfd.write(f"{word} ")
                if count_words:
                    word_count.update(words)
                os.remove(dir_name + f)
            except UnicodeDecodeError:
                # We only look at utf-8 formatted email.
                os.remove(dir_name + f)
    if count_words:
        return {word: cnt for word, cnt in word_count.items() if cnt >=
                treshold}


def preprocess_file(filename):
    with open(filename, "r") as rfd:
        try:
            data = email.parser.Parser().parse(rfd)
            body = data
            while body.is_multipart():
                body = body.get_payload()[0]
            body = body.get_payload()
            return parse_email(body)
        except UnicodeDecodeError:
            return None


# Necessary nltk package for tokenization.
nltk.download('punkt', quiet=True)
