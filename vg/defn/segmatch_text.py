import numpy
import numpy as np
import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from onion import rhn, attention, conv
from vg.simple_data import vector_padder
from vg.defn.encoder_em import Encoder
import vg.scorer
from vg.scorer import Scorer, testing
from collections import Counter
import sys
from onion.loss import cosine_matrix, contrastive 
import json

def step(task, *args):
    loss = task.train_cost(*args)
    task.optimizer.zero_grad()
    loss.backward()
    _ = nn.utils.clip_grad_norm(task.parameters(), task.config['segmatch']['max_norm'])
    return loss


class Segmatch(nn.Module):

    def __init__(self, config):
        super(Segmatch, self).__init__()
        util.autoassign(locals())
        self.Encode = Encoder(**config['encoder'])        
        self.ProjBeg = nn.Linear(config['segmatch']['size'], config['segmatch']['size_target'])
        self.ProjEnd = nn.Linear(config['segmatch']['size'], config['segmatch']['size_target'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['segmatch']['lr'])

    def step(self, *args):
        return step(self, *args)

    def score(self, x, y):
        return F.cosine_similarity(x, y, dim=1)
        
    def forward(self, speech):
        return F.normalize(self.ProjBeg(self.Encode(speech)), p=2, dim=1)

    def cost(self, beg, end):
        beg_encoded = F.normalize(self.ProjBeg(self.Encode(beg)), p=2, dim=1)
        end_encoded = F.normalize(self.ProjEnd(self.Encode(end)), p=2, dim=1)
        scores = cosine_matrix(beg_encoded, end_encoded) 
        return contrastive(scores, margin=self.config['segmatch']['margin_size'])

    def train_cost(self, beg, end):
        return self.cost(beg, end)

    def test_cost(self, beg, end):
        with testing(self):
            self.eval()
            cost = self.cost(beg, end)
        return cost

    def predict(self, speech):
        with testing(self):
            self.eval()
            pred = self(speech)
        return pred

    def args(self, item):
        beg = item['input_beg'].astype('int64')
        end = item['input_end'].astype('int64')
        assert min(beg.shape) > 0 and min(end.shape) > 0, "Broken input {}".format(item['input'])
        return (beg, end)

def experiment(net, data, prov, model_config, run_config):
    def valid_loss(task):
        result = []
        for item in data.iter_valid_batches():
            args = task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append(task.test_cost(*args).data.cpu().numpy())
        return result
    
    net.cuda()
    net.train()
    scorer = Scorer(data.provider, 
                    dict(split='val', 
                         tokenize=data.tokenize, 
                         batch_size=data.batch_size,
                         encode_sentences=vg.scorer.encode_texts))
    net.optimizer.zero_grad()
    last_epoch = 0
    with open("result.json", "w") as out:

      for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        costs = Counter()
        net.train()
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                name = "Segmatch"; task = net
                spk = item['speaker'][0] if len(set(item['speaker'])) == 1 else 'MIXED'
                args = task.args(item)
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
                loss = task.optimizer.step(lambda: task.step(*args))
                costs += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data.batch_size, name, spk, "train", "".join([str(costs['cost']/costs['N'])]))
                if j % run_config['validate_period'] == 0:
                    print(epoch, j, 0, name, spk, "valid", "".join([str(numpy.mean(valid_loss(task)))]))
                  #  with testing(net):
                  #      result = dict(epoch=epoch, 
                  #            rsa=scorer.rsa_image(net), 
                  #            retrieval_para=scorer.retrieval_para(net))
                  #      out.write(json.dumps(result))
                  #      out.write("\n")
                  #      out.flush()

                sys.stdout.flush()
        torch.save(net, "model.{}.pkl".format(epoch))
        with testing(net):
            result = dict(epoch=epoch, 
                          rsa=scorer.rsa_image(net), 
                          retrieval_para=scorer.retrieval_para(net))
            out.write(json.dumps(result))
            out.write("\n")
            out.flush()

    torch.save(net, "model.pkl")



