import os
import numpy as np
import json
import sys
import gzip

def parse_map(lines):
    M = {}
    for line in lines:
        fields  =line.split()
        M[fields[0]] = ' '.join(fields[1:])
    return M

class Provider:

  def __init__(self, dataset, root='.', audio_kind='mfcc', truncate=None):
    assert dataset == 'places'
    assert audio_kind == 'mfcc'
    self.root = root
    self.dataset_name = dataset
    self.audio_kind = audio_kind
    self.truncate = truncate    
    self.audiofile = "{}/data/places/places_mfcc.npy".format(self.root)
    self.imagefile = "{}/data/places/placeimgsfeatsvgg19.npy".format(self.root)
    
    self.utt2ASR = parse_map(open("{}/data/places/metadata/utt2ASR".format(self.root), encoding='utf-8'))
    self.utt2wav = parse_map(open("{}/data/places/metadata/utt2wav".format(self.root), encoding='utf-8'))
    self.utt2image = parse_map(open("{}/data/places/metadata/utt2image".format(self.root), encoding='utf-8'))
    self.utt2spk = parse_map(open("{}/data/places/metadata/utt2spk".format(self.root), encoding='utf-8'))
    self.id = dict(train = set(line.strip() for line in open("{}/data/places/lists/acl_2017_train_uttids".format(self.root))),
                   val = set(line.strip() for line in open("{}/data/places/lists/acl_2017_val_uttids".format(self.root))))
    self.audio = np.load(self.audiofile).item(0)
    self.imageid = [ line.split()[1] for line in open("{}/data/places/metadata/utt2image".format(self.root), encoding='utf-8') ]
    self.images = dict((imgid, feat) for imgid, feat in zip(self.imageid, np.load(self.imagefile)))
   
  def iterImages(self, split='train', shuffle=False):

    for sent in self.iterSentences(split=split, shuffle=shuffle):
        image = dict(sentences=[sent], imgid=sent['imgid'], feat=self.images[sent['imgid']])
        yield image

  def iterSentences(self, split='train', shuffle=False):
    if shuffle:
        ID = sorted(self.id[split], key=lambda _: np.random.random())
    else:
        ID = self.id[split]
    for uttid in ID:
        sent = dict(tokens = self.utt2ASR[uttid].split(), 
                    raw = self.utt2ASR[uttid],
                    imgid = self.utt2image[uttid],
                    audio = self.audio[uttid][:self.truncate,:] if self.truncate is not None 
                            else self.audio[uttid],
                    sentid=uttid,
                    speaker="places_" + self.utt2spk[uttid])
        yield sent



def getDataProvider(*args, **kwargs):
	return Provider(*args, **kwargs)
