import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np

ids, txts, labels = [], [], []
with open("./data/semanticf8k/semantic_flickraudio_labels.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    reader.next()
    for row in reader:
        cocoid, txt, label = row[0], row[1],  row[2].split("|")
        ids.append(cocoid)
        txts.append(txt)
        labels.append(label)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(txts)
binarizer = MultiLabelBinarizer()
y = binarizer.fit_transform(labels)

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = OneVsRestClassifier(Perceptron())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.average_precision_score(y_pred, y_test))
