import argparse
import logging
import vg.bundle as bundle
import random
import numpy
import onion.util as util
import torch
from vg.simple_data import vector_padder
from sklearn.metrics.pairwise import cosine_similarity
from vg.evaluate import ranking, paraphrase_ranking
from collections import Counter


def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()

    nn_p = commands.add_parser('nn')
    nn_p.set_defaults(func=nn)
    nn_p.add_argument('model',  help='Model file(s)')
    nn_p.add_argument('--dataset', default='coco')
    nn_p.add_argument('--split', default='val')
    nn_p.add_argument('--speaker', default=None)
    nn_p.add_argument('--N', default=100, type=int)
    nn_p.add_argument('--max', default=5000, type=int)
    nn_p.add_argument('--seed', default=None, type=int)
    nn_p.add_argument('--root', default='.')
    nn_p.add_argument('--batch_size', default=32, type=int)
    nn_p.add_argument('--classic', action='store_true', help="Classic version of data provider")

    retrieve_p = commands.add_parser('retrieve')
    retrieve_p.set_defaults(func=retrieve)
    retrieve_p.add_argument('model',  nargs='+', help='Model file(s)')
    retrieve_p.add_argument('--dataset', default='flickr8k')
    retrieve_p.add_argument('--split', default='val')
    retrieve_p.add_argument('--root', default='.')
    retrieve_p.add_argument('--batch_size', default=32, type=int)

    retrieve_para_p = commands.add_parser('retrieve_para')
    retrieve_para_p.set_defaults(func=retrieve_para)
    retrieve_para_p.add_argument('model',  nargs='+', help='Model file(s)')
    retrieve_para_p.add_argument('--dataset', default='flickr8k')
    retrieve_para_p.add_argument('--split', default='val')
    retrieve_para_p.add_argument('--root', default='.')
    retrieve_para_p.add_argument('--batch_size', default=32, type=int)

    rsa_p = commands.add_parser('rsa')
    rsa_p.set_defaults(func=rsa)
    rsa_p.add_argument('model',  nargs='+', help='Model file(s)')
    rsa_p.add_argument('--dataset', default='flickr8k')
    rsa_p.add_argument('--split', default='val')
    rsa_p.add_argument('--root', default='.')
    rsa_p.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()
    args.func(args)    

def rsa(args):
    from vg.scorer import Scorer
    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    config = dict(split=args.split, tokenize=lambda x: x['audio'], encode_sentences=encode_sentences, 
                  batch_size=args.batch_size)
    scorer = Scorer(prov, config)
    for path in args.model:
        task = load(path)
        task.eval().cuda()
        print(path, scorer.rsa_image(task))

def retrieve(args):

    def encode_images(task, imgs, batch_size=128):
        """Project imgs to the joint space using model.
        """
        return numpy.vstack([ task.Visual.encode_images(torch.autograd.Variable(torch.from_numpy(numpy.vstack(batch))).cuda()).data.cpu().numpy()
                          for batch in util.grouper(imgs, batch_size) ])


    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    sent = [ s for s in prov.iterSentences(split=args.split) ]
    audio = [ s['audio'] for s in sent ]
    logging.info('Embedding utterances')
    images = list(prov.iterImages(split=args.split))
    correct_img = numpy.array([ [ sent[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sent)) ] )
    for path in args.model:
        #logging.info('Loading model')
        task = torch.load(path)
        task.eval().cuda()
        img_fs = encode_images(task, [ img['feat'] for img in images ])
        pred = encode_sentences(task, audio, batch_size=args.batch_size)
        data = ranking(img_fs, pred, correct_img, ns=(1,5,10), exclude_self=False)
        recall = data['recall']
        print (path, 
                  numpy.mean(recall[1]),\
                  numpy.mean(recall[5]),\
                  numpy.mean(recall[10]),\
                  numpy.median(data['ranks']))


def retrieve_para(args):

    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    sent = [ s for s in prov.iterSentences(split=args.split) ]
    audio = [ s['audio'] for s in sent ]
    logging.info('Embedding utterances')
    correct_para = numpy.array([ [ sent[i]['imgid']==sent[j]['imgid']
                                      for j in range(len(sent)) ]
                                    for i in range(len(sent)) ] )
    for path in args.model:
        #logging.info('Loading model')
        task = torch.load(path)
        task.eval().cuda()
        pred = encode_sentences(task, audio, batch_size=args.batch_size)
        data = paraphrase_ranking(pred, correct_para, ns=(1,5,10))
        recall = data['recall']
        print (path, 
                  numpy.mean(recall[1]),\
                  numpy.mean(recall[5]),\
                  numpy.mean(recall[10]),\
                  numpy.median(data['ranks']))
def nn(args):
    if args.seed is not None:
        random.seed(args.seed)
    #model = bundle.load(args.model)
    task = load(args.model)
    audio_kind = 'mfcc'
    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k' and args.classic:
        import vg.data_provider as dp
        audio_kind = 'human.max1K.accel3.ord.mfcc'
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    elif args.dataset == 'libri':
        import vg.libri_provider as dp
    else:
        raise ValueError
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind=audio_kind)
    logging.info('Loading model')
    #model.task.eval().cuda()
    task.eval().cuda()
    sent = [ s for s in prov.iterSentences(split=args.split) \
                if args.speaker is None or args.speaker == s['speaker'] ][:args.max]
    audio = [ s['audio'] for s in sent ]
    logging.info('Embedding utterances')
    #pred = encode_sentences_old(model, audio, batch_size=args.batch_size)
    pred = encode_sentences(task, audio, batch_size=args.batch_size)
    logging.info('Embedded sentences: {}'.format(pred.shape))
    logging.info('Computing similarity')
    sim = cosine_similarity(pred)
    logging.info('Printing sample')
    same = 0
    tot = 0
    tally = Counter()
    for i in random.sample(range(len(sent)), args.N):
        match = sent[sim[i].argsort()[-2]]
        print(sent[i].get('speaker'), sent[i]['raw'])
        print(match.get('speaker'), match['raw'])
        print()
        tot += 1
        if sent[i].get('speaker') == match.get('speaker'):
            same += 1
        tally += Counter([sent[i].get('speaker')])
    count = numpy.array([ val for key, val in tally.items() ])
    prop = count/count.sum()
    print("Expected same: ", prop.dot(prop))
    print("Seen same:     ", same / tot)
    
def load(path):
    try:
        model = bundle.load(path)
        return model.task
    except:
        task = torch.load(path)
        return task

def encode_sentences_old(model, audios, batch_size=128):

    return numpy.vstack([ model.task.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                vector_padder(batch))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(audios, batch_size) ])

def encode_sentences(task, audios, batch_size=128):

    return numpy.vstack([ task.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                vector_padder(batch))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(audios, batch_size) ])
    



if __name__ == '__main__':
    main()

