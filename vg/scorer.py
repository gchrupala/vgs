import random
import numpy
#random.seed(123)
#numpy.random.seed(123)
from sklearn.metrics.pairwise import cosine_similarity
from vg.evaluate import ranking, paraphrase_ranking
import scipy
from vg.simple_data import vector_padder
import torch
import onion.util as util
import contextlib
import argparse
import logging
import vg.bundle as bundle
import pprint
import itertools
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from vg.simple_data import characters
import json

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()

    p = commands.add_parser('score')
    p.set_defaults(func=score)
    p.add_argument('model',  nargs='+', help='Model file(s)')
    p.add_argument('--text', action='store_true')
    p.add_argument('--dataset', default='flickr8k')
    p.add_argument('--split', default='val')
    p.add_argument('--root', default='.')
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--output', default='result.json')
    args = parser.parse_args()
    args.func(args)  

def score(args):
    
    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    tokenize = characters if args.text else lambda x: x['audio']
    config = dict(split=args.split, tokenize=tokenize, batch_size=args.batch_size)
    if args.text:
        config['encode_sentences'] = encode_texts
    scorer = Scorer(prov, config)
    output = []
    for path in args.model:
        task = load(path)
        task.eval().cuda()
        rsa = scorer.rsa_image(task)
        para = scorer.retrieval_para(task)
        result = dict(path=path, rsa=rsa, para=para)
        if not args.text:
            result['speaker_id'] = scorer.speaker_id(task)
        output.append(result)
    json.dump(output, open(args.output, 'w'), indent=2)

def load(path):
    try:
        model = bundle.load(path)
        return model.task
    except:
        task = torch.load(path)
        return task

@contextlib.contextmanager
def testing(net):
    mode = net.training
    net.eval()
    yield net
    net.training = mode


class Scorer:

    def __init__(self, prov, config, net=None):
        self.prov = prov
        self.config = config
        self.sentences = []
        self.rsa_image_data = []
        self.images = []
        self.encode_sentences = config.get('encode_sentences', encode_sentences)
        self.encode_images = config.get('encode_images', encode_images)
        for image in prov.iterImages(split=config['split']):
           self.images.append(image)
           for sent in image['sentences']:
               self.rsa_image_data.append(image['feat'])
               self.sentences.append(sent)
        self.sentence_data = [ config['tokenize'](s) for s in self.sentences ]
        self.sim_images = cosine_similarity(self.rsa_image_data)
        self.correct_para = numpy.array([ [ self.sentences[i]['imgid']==self.sentences[j]['imgid']
                                      for j in range(len(self.sentences)) ]
                                    for i in range(len(self.sentences)) ] )
        self.correct_img = numpy.array([ [ self.sentences[i]['imgid']==self.images[j]['imgid']
                                          for j in range(len(self.images)) ]
                                        for i in range(len(self.sentences)) ] )
        self.net = net
        if self.net is not None:
            self.pred = self.encode_sentences(self.net, self.sentence_data, batch_size=self.config['batch_size'])
        self.speakers = Counter(s['speaker'] for s in self.sentences)

    def speaker_id(self, net=None):
        if net is None:
            pred = self.pred
            net = self.net
        else: 
            with testing(net):
                pred = self.encode_sentences(net, self.sentence_data, batch_size=self.config['batch_size'])
        X = pred
        if hasattr(net, 'mapper'):
            # FIXME do something reasonable here
            Z = numpy.array([ numpy.zeros((1)) for audio in self.sentence_data])
        else:
            Z = numpy.array([ audio.mean(axis=0) for audio in self.sentence_data])
        y = LabelEncoder().fit_transform([s['speaker'] for s in self.sentences])
        C = [ 10**p for p in range(2, 3) ]
        X, X_test, Z, Z_test, y, y_test = train_test_split(X, Z, y, random_state=42)
        scores = dict(rep=[], mfcc=[])
        counts = numpy.array(list(self.speakers.values()))
        maj = counts.max()/counts.sum()
        for c in C:
            model_rep = LogisticRegression(C=c)
            model_rep.fit(X, y)
            acc_rep = model_rep.score(X_test, y_test)
            model_mfcc = LogisticRegression(C=c)
            model_mfcc.fit(Z, y)
            acc_mfcc = model_mfcc.score(Z_test, y_test)
            logging.info("speaker_id acc {} {} {} {}".format(c, acc_rep, acc_mfcc, rer(acc_mfcc, acc_rep)))
            scores['rep'].append(acc_rep)
            scores['mfcc'].append(acc_mfcc)
        return dict(maj=maj, rep=max(scores['rep']), mfcc=max(scores['mfcc']))

    def rsa_image(self, net=None, within=False):
            # Full RSA
            if net is None:
                pred = self.pred
            else:
                with testing(net):
                   pred = self.encode_sentences(net, self.sentence_data, batch_size=self.config['batch_size'])
            if hasattr(net, 'mapper') and net.mapper is not None:
                # FIXME do something reasonable here
                #print("This is a text net")
                mfcc = numpy.array([ numpy.zeros((1)) for audio in self.sentence_data])
            else:
                #print("This is an audio net")
                mfcc = numpy.array([ audio.mean(axis=0) for audio in self.sentence_data])
            sim_mfcc = cosine_similarity(mfcc)
            sim_pred = cosine_similarity(pred)
            
            img_rep = scipy.stats.pearsonr(triu(self.sim_images), triu(sim_pred))[0]
            img_mfcc = scipy.stats.pearsonr(triu(self.sim_images), triu(sim_mfcc))[0]
            rep_mfcc = scipy.stats.pearsonr(triu(sim_pred), triu(sim_mfcc))[0]
            result = dict(img_rep=float(img_rep), img_mfcc=float(img_mfcc), rep_mfcc=float(rep_mfcc)) # make json happy
            if within:
                result['within'] = within_rsa
            return result

    def retrieval(self, net=None):
            img_fs = self.encode_images(net, [ s['feat'] for s in self.images ])
            if net is None:
                pred = self.pred
            else:
                with testing(net):
                    pred = self.encode_sentences(net, self.sentence_data, batch_size=self.config['batch_size'])

            result = {}
            ret = ranking(img_fs, pred, self.correct_img, ns=(1,5,10), exclude_self=False)
            result['recall@1'] = numpy.mean(ret['recall'][1])
            result['recall@5'] = numpy.mean(ret['recall'][5])
            result['recall@10'] = numpy.mean(ret['recall'][10])
            result['medr'] = numpy.median(ret['ranks'])
            return result
    
    def retrieval_para(self, net=None):
            if net is None:
                pred = self.pred
            else:
                with testing(net):
                   pred = self.encode_sentences(net, self.sentence_data, batch_size=self.config['batch_size'])

            result = {}
            ret = paraphrase_ranking(pred, self.correct_para, ns=(1,5,10))
            result['recall@1'] = numpy.mean(ret['recall'][1])
            result['recall@5'] = numpy.mean(ret['recall'][5])
            result['recall@10'] = numpy.mean(ret['recall'][10])
            result['medr'] = numpy.median(ret['ranks'])
            return result
    
def rer(hi, lo):
    return ((1-lo)-(1-hi))/(1-lo)

def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = numpy.ones_like(x)
    return x[numpy.triu(ones, k=1) == 1]

def encode_sentences(task, audios, batch_size=128):
    return numpy.vstack([ task.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                vector_padder(batch))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(audios, batch_size) ])

def encode_texts(task, texts, batch_size=128):
    return numpy.vstack([ task.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                task.batcher.batch_inp(task.mapper.transform(batch)).astype('int64'))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(texts, batch_size) ])

def encode_images(task, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ task.encode_images(
                            torch.autograd.Variable(torch.from_numpy(
                                numpy.vstack(batch))).cuda()).data.cpu().numpy()
                          for batch in util.grouper(imgs, batch_size) ])




if __name__ == '__main__':
    main()

