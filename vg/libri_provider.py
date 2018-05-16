import os
import numpy as np
import json
import sys
import gzip
import glob
from functools import reduce
import csv
import logging

from sklearn.utils import resample 
def load_mfcc(path):
    L = (np.load(f).item(0) for f in glob.glob(path))
    return reduce(lambda x, z: {**x, **z}, L)

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', truncate=None):
    assert dataset == 'libri'
    assert audio_kind == 'mfcc'
    self.root = root
    self.dataset_name = dataset
    self.audio_kind = audio_kind
    self.truncate = truncate    
    self.audiopath = "{}/data/libri/mfcc/libri_*.npy".format(self.root)
    self.metapath = "{}/data/libri/libri.csv".format(self.root)
    self.audio = load_mfcc(self.audiopath)
    meta_all = dict([(row[0].split("/")[-1], row[1:]) for row in csv.reader(open("{}/data/libri/libri.csv".format(self.root)), delimiter='\t') ][1:])
    self.meta = dict((k,v) for k,v in meta_all.items() if k in self.audio)
    self.val = set(resample(list(self.meta.keys()), replace=False, random_state=123)[:1000])
    logging.info("{} items in libri.csv, {} items in mfcc".format(len(meta_all), len(self.audio)))    

   
  def iterSentences(self, split='train', shuffle=False):
    assert split in ['train', 'val']
    if shuffle:
        ID = sorted(self.meta.keys(), key=lambda _: np.random.random())
    else:
        ID = self.meta.keys()
    for uttid in ID:
        if (split == 'val' and uttid in self.val) or (split == 'train' and uttid not in self.val):
            
                sent = dict(tokens = self.meta[uttid][2].split(), 
                    raw =    self.meta[uttid][2],
                    audio =  self.audio[uttid][:self.truncate,:] if self.truncate is not None 
                                                                   else self.audio[uttid],
                    sentid=uttid,
                    chapter=self.meta[uttid][1],
                    speaker="libri_" + self.meta[uttid][0])
                yield sent

  def iterImages(self, split='train', shuffle=False):
    # For compatibility. There are no images in this dataset
    for sent in self.iterSentences(split=split, shuffle=shuffle):
        yield dict(sentences=[sent])

def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)

