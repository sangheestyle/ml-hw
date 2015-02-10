from csv import DictReader, DictWriter

import random
import numpy as np
from numpy import array

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


random.seed(random.random())


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if not t.isdigit()]


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english',
                              ngram_range=(1,2),
                              tokenizer=LemmaTokenizer())

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])

    random.shuffle(train)
    l = len(train)

    x_train = feat.train_feature(x['text'] for x in train[:-5000])

    x_test_in = feat.test_feature(x['text'] for x in train[-5000:l])
    x_test = feat.test_feature(x['text'] for x in test)

    y_train = array(list(labels.index(x['cat']) for x in train[:-5000]))
    y_test_in = array(list(labels.index(x['cat']) for x in train[-5000:l]))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions_in = lr.predict(x_test_in)

    print("Accuracy: %f" % accuracy_score(y_test_in, predictions_in))

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
