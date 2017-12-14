import numpy
import random
import vg.simple_data as sd
import vg.bundle as bundle
import vg.evaluate as evaluate
from collections import Counter
import json
import onion.util as util
import torch
import torch.nn as nn
import torch.optim as optim
import sys

def run_train(data, prov, model_config, run_config, eval_config, runid='', resume=False):

    seed  = run_config.get('seed')
    epoch_evals = []
    if  seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
    if resume:
        raise NotImplementedError
    else:
        last_epoch = 0
        model = bundle.GenericBundle(dict(scaler=data.scaler,
                                           batcher=data.batcher), model_config, run_config['task'])
    model.task.cuda()
    model.task.train()

    print("Params: {}".format(sum([ numpy.prod(param.size()) for param in model.task.parameters() ])))
    for name, param in model.task.named_parameters():
        print(name, param.size(), param.requires_grad)
    def epoch_eval():
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
        return evaluate.ranking(img_fs, predictions, correct_img, ns=(1,5,10), exclude_self=False)

    def valid_loss():
        result = []
        for item in data.iter_valid_batches():
            args = model.task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append(model.task.test_cost(*args).data.cpu().numpy())
        return result

    costs = Counter()

    optimizer = optim.Adam(model.task.parameters(), lr=model.task.config['lr'])
    for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        model.task.train()
        random.shuffle(data.data['train'])
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                args = model.task.args(item)
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
                loss = model.task.train_cost(*args)
                loss.backward()
                prenorm = nn.utils.clip_grad_norm(model.task.parameters(), model.task.max_norm)
                optimizer.step()
                costs += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data.batch_size, "train", "".join([str(costs['cost']/costs['N'])]))
                if j % run_config['validate_period'] == 0:
                        print(epoch, j, 0, "valid", "".join([str(numpy.mean(valid_loss()))]))
                sys.stdout.flush()
        model.save(path='model.r{}.e{}.zip'.format(runid,epoch))
        #epoch_evals.append(epoch_eval())
        #json.dump(epoch_evals[-1], open('scores.{}.json'.format(epoch),'w'))
    model.save(path='model.r{}.zip'.format(runid))
    return epoch_evals


def norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
