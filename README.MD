# svm-spam-filter

A python implementation of a simple spam classifier using support-vector
machines. Requires python3.6+ to run.

## Usage

First obtain enough emails to train the classifier. For example, use the
SpamAssassin public mail corpus: https://spamassassin.apache.org/old/publiccorpus/.
Place spam in a folder called spam and non-spam in a folder called ham. Then,
train the classifier:

``
python3 main.py
``

Note that preprocessing of the emails might take a while, depending on the amount of
the emails that are used to train the classifier.
To classify your own emails (after you have trained the classifier), run the following:

``
python3 predict.py FILE
``

Where FILE contains the source code of the email you want to classify.

