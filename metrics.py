import argparse
import logging
import vg.bundle as bundle
import random
import numpy
import onion.util as util
import torch
from vg.simple_data import vector_padder
from sklearn.metrics.pairwise import cosine_similarity

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()

    nn_p = commands.add_parser('nn')
    nn_p.set_defaults(func=nn)
    nn_p.add_argument('model',  help='Model file')
    nn_p.add_argument('--dataset', default='coco')
    nn_p.add_argument('--split', default='val')
    nn_p.add_argument('--speaker', default=None)
    nn_p.add_argument('--N', default=20, type=int)
    nn_p.add_argument('--max', default=5000, type=int)
    nn_p.add_argument('--seed', default=None, type=int)
    nn_p.add_argument('--root', default='.')
    nn_p.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    args.func(args)    

def nn(args):
    if args.seed is not None:
        random.seed(args.seed)
    model = bundle.load(args.model)
    if args.dataset == 'coco':
        import vg.vendrov_provider as dp
    elif args.dataset == 'places':
        import vg.places_provider as dp
    elif args.dataset == 'flickr8k':
        import vg.flickr8k_provider as dp
    logging.info('Loading data')
    prov = dp.getDataProvider(args.dataset, root=args.root, audio_kind='mfcc')
    logging.info('Loading model')
    model.task.eval().cuda()
    sent = [ s for s in prov.iterSentences(split=args.split) \
                if args.speaker is None or args.speaker == s['speaker'] ][:args.max]
    audio = [ s['audio'] for s in sent ]
    logging.info('Embedding utterances')
    pred = encode_sentences(model, audio, batch_size=args.batch_size)
    logging.info('Computing similarity')
    sim = cosine_similarity(pred)
    logging.info('Printing sample')
    for i in random.sample(range(len(sent)), args.N):
        match = sent[sim[i].argsort()[-2]]
        print(sent[i]['speaker'], sent[i]['raw'])
        print(match['speaker'], match['raw'])
        print()


def encode_sentences(model, audios, batch_size=128):

    return numpy.vstack([ model.task.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                vector_padder(batch))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(audios, batch_size) ])



if __name__ == '__main__':
    main()

