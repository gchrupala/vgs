import numpy
import random
import vg.simple_data as sd
from vg.simple_data import words
import vg.bundle as bundle
from vg.evaluate import ranking, paraphrase_ranking
from collections import Counter
import json
import onion.util as util
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

def run_train(data, prov, model_config, run_config, eval_config, runid='', resume=False):

    if resume:
        raise NotImplementedError
    else:
        last_epoch = 0
        model = bundle.GenericBundle(dict(scaler=data.scaler,
                                           batcher=data.batcher), model_config, run_config['task'])
    model.task.cuda()
    model.task.train()

    print("Params: {}".format(sum([ numpy.prod(param.size()) for param in model.task.parameters() ])))
    #for name, param in model.task.named_parameters():
    #    print(name, param.size(), param.requires_grad)

    if eval_config['mode'] == 'corr':
        #train_text = (s['raw'] for s in prov.iterSentences(split='train'))
        split_text = (s['raw'] for s in prov.iterSentences(split=eval_config['split']))
        vec = TfidfVectorizer(analyzer='word')
        #vec.fit_transform(train_text)
        split_vecs = vec.fit_transform(split_text)        
        SIM_tfidf = cosine_similarity(split_vecs)        

    def epoch_eval_corr():
        model.task.eval()
        scaler = model.scaler
        batcher = model.batcher
        mapper = batcher.mapper
        sents = list(prov.iterSentences(split=eval_config['split']))
        sents_tok =  [ eval_config['tokenize'](sent) for sent in sents ]
        predictions = eval_config['encode_sentences'](model, sents_tok, batch_size=eval_config['batch_size'])
        SIM_pred = cosine_similarity(predictions)
        result = scipy.stats.spearmanr(SIM_tfidf.flatten(), SIM_pred.flatten()) #FIXME only take upper triangular
        model.task.train() 
        return result

    def epoch_eval_para():
        model.task.eval()
        scaler = model.scaler
        batcher = model.batcher
        mapper = batcher.mapper
        sents = list(prov.iterSentences(split=eval_config['split']))
        sents_tok =  [ eval_config['tokenize'](sent) for sent in sents ]
        predictions = eval_config['encode_sentences'](model, sents_tok, batch_size=eval_config['batch_size'])
        correct_para = numpy.array([ [ sents[i]['imgid']==sents[j]['imgid']
                                      for j in range(len(sents)) ]
                                    for i in range(len(sents)) ] )
        result = paraphrase_ranking(predictions, correct_para)
        model.task.train()
        return result

    def epoch_eval(mode='image', para=False):
        if para: # compatibility 
            mode = 'para'
        if mode == 'para':
            return epoch_eval_para()
        elif mode == 'corr':
            return epoch_eval_corr()
        # otherwise image
        model.task.eval()
        task = model.task
        scaler = model.scaler
        batcher = model.batcher
        mapper = batcher.mapper
        sents = list(prov.iterSentences(split=eval_config['split']))
        sents_tok =  [ eval_config['tokenize'](sent) for sent in sents ]
        predictions = eval_config['encode_sentences'](model, sents_tok, batch_size=eval_config['batch_size'])
        images = list(prov.iterImages(split=eval_config['split']))
        img_fs = imaginet.task.encode_images(model, [ img['feat'] for img in images ])
        correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                      for j in range(len(images)) ]
                                    for i in range(len(sents)) ] )
        result = ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)
        model.task.train()
        return result

    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            args = model.task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append(model.task.test_cost(*args).data.cpu().numpy())
        return result

    costs = Counter()

    optimizer = optim.Adam(model.task.parameters(), lr=model.task.config['lr'])
    optimizer.zero_grad()
    epoch_evals = []
    for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        model.task.train()
        random.shuffle(data.data['train'])
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                args = model.task.args(item)
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
                loss = model.task.train_cost(*args)
                optimizer.zero_grad()
                loss.backward()
                _ = nn.utils.clip_grad_norm(model.task.parameters(), model.task.max_norm)
                optimizer.step()
                costs += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])]))
                if j % run_config['validate_period'] == 0:
                        print(epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))]))
                        epoch_evals.append(epoch_eval(mode=eval_config.get('mode', 'image'), para=eval_config.get('para', False)))
                        numpy.save('scores.{}.{}.npy'.format(epoch,j), epoch_evals[-1])
                sys.stdout.flush()
                # FIXME just testing
        model.save(path='model.r{}.e{}.zip'.format(runid,epoch))
        epoch_evals.append(epoch_eval(mode=eval_config.get('mode', 'image'), para=eval_config.get('para', False)))
        #json.dump(epoch_evals[-1], open('scores.{}.json'.format(epoch),'w'))
        numpy.save('scores.{}.npy'.format(epoch), epoch_evals[-1])

    model.save(path='model.r{}.zip'.format(runid))
#    return epoch_evals

def run_eval(prov, config, encode_sentences=None, encode_images=None, start_epoch=1, runid=''):
    datapath='../..'
    for epoch in range(start_epoch, 1+config['epochs']):
        scores = evaluate(prov,
                          datapath=datapath,
                          tokenize=config['tokenize'],
                          split=config['split'],
                          task=config['task'],
                          encode_sentences=encode_sentences,
                          encode_images=encode_images,
                          batch_size=config['batch_size'],
                          model_path='model.r{}.e{}.zip'.format(runid, epoch))
        numpy.save('scores.{}.npy'.format(epoch), scores)
        #json.dump(scores, open('scores.{}.json'.format(epoch),'w'))

def evaluate(prov,
             datapath='.',
             model_path='model.zip',
             batch_size=128,
             task=None,
             encode_sentences=None,
             encode_images=None,
             tokenize=words,
             split='val'
            ):
    model = bundle.load(path=model_path)
    model.task.cuda()
    model.task.eval()
    scaler = model.scaler
    batcher = model.batcher
    mapper = batcher.mapper
    sents = list(prov.iterSentences(split=split))
    sents_tok =  [ tokenize(sent) for sent in sents ]
    predictions = encode_sentences(model, sents_tok, batch_size=batch_size) #FIXME this needs to be converted to Variables
    images = list(prov.iterImages(split=split))
    img_fs = encode_images(model, [ img['feat'] for img in images ])
    #img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct_img = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                                  for j in range(len(images)) ]
                                for i in range(len(sents)) ] )
    return ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)

def norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
