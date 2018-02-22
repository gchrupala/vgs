import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics


ids, txts, labels = [], [], []
with open("/roaming/u1257964/semanticf8k/semantic_flickraudio_labels.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    reader.next()
    for row in reader:
        cocoid, txt, label = row[0], row[1],  row[2].split("|")
        ids.append(cocoid)
        txts.append(txt)
        labels.append(label)

f8k_root = "/exp/gchrupal/corpora/flickr_audio/wavs/"

D_mfcc= np.load("/roaming/u1257964/semanticf8k/sf8kmfcc.npy").item()

X_mfcc_sum = []
X_mfcc_mean = []
X_mfcc_max = []
for cocoid in ids:
    path = os.path.join(f8k_root, cocoid + ".wav")
    feat = D_mfcc[path]
    X_mfcc_sum.append(np.sum(feat, axis=0))
    X_mfcc_mean.append(np.mean(feat, axis=0))
    X_mfcc_max.append(np.max(feat, axis=0))
X_mfcc_sum = np.array(X_mfcc_sum)
X_mfcc_mean = np.array(X_mfcc_mean)
X_mfcc_max = np.array(X_mfcc_max)

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(txts)
datasets = [X_text, X_mfcc_sum, X_mfcc_mean, X_mfcc_max]
# for i, j in enumerate(datasets[1:]):
#    datasets[i] = np.array(j)


names = ["text", "sum mfcc", "mean mfcc", "max mfcc"]

binarizer = MultiLabelBinarizer()
y = binarizer.fit_transform(labels)

for name, X in zip(names, datasets):
    print("Training on {}".format(name))
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clf = OneVsRestClassifier(Perceptron())
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(metrics.f1_score(y_pred, y_test, average="macro"))
