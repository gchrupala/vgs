import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from torch import nn
from torch import optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, dropout=0.5, 
                 max_epoch=50, batch_size=64, patience=2):
        """
        input_dim: int, dimensionality of the input
        hidden_dim: int, hidden size
        n_out: int, number of output classes
        dropout: float, dropout probability
        max_epoch: int, maximum number of epochs to train
        batch_size, int, batch size
        patience: int, stop after patience number of epochs of no improvement
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = n_out
        self.dropout = dropout
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                               nn.Dropout(p=self.dropout),
                               nn.Sigmoid(),
                               nn.Linear(self.hidden_dim, self.n_out)).cuda()
        self.patience = patience
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.all_costs = []

    def gen_split(self, X, y):
        permutation = np.random.permutation(len(X))
        X_train, X_test, y_train, y_test = train_test_split(X[permutation], 
                                                            y[permutation])
        return X_train, X_test, y_train, y_test
    
    def train_epoch(self, X, y, epoch_size=1):
        self.model.train()
        losses = []
        for i in range(0, len(X), self.batch_size):
            # forward
            idx = torch.LongTensor(permutation[i:i + self.batch_size]).cuda()
            Xbatch = Variable(X.index_select(0, idx)).cuda()
            ybatch = Variable(y.index_select(0, idx)).cuda()
            output = self.model(Xbatch)
            # loss
            loss = self.criterion(output, ybatch)
            losses.append(loss.data.mean())
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss)
        self.nepoch += 1
        print('[%d/%d] Loss: %.3f' % (self.nepoch, self.max_epochs, np.mean(losses)))  
    
    def eval(self, X, y):
        self.model.eval()
        costs = []
        for i in range(0, len(X), self.batch_size):
            # forward
            idx = torch.LongTensor(permutation[i:i + self.batch_size]).cuda()
            Xbatch = Variable(X.index_select(0, idx)).cuda()
            ybatch = Variable(y.index_select(0, idx)).cuda()
            output = self.model(Xbatch)
            # loss
            loss = self.criterion(output, ybatch)
            costs.append(loss.data.mean())
            # backward
           print(loss)
        print(np.mean(losses))

    def fit(self, X, y, early_stop=True):
        self.nepoch = 0
        best = -2
        stop_train = False
        early_stop_count = 0
        X_train, X_test, y_train, y_test = self.gen_split(X, y)
        for i in range(self.max_epoch):
            train_epoch(X_train, y_train)

            

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
