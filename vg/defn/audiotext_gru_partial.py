# Pytorch version of imaginet.audiovis_rhn
import numpy
import numpy as np
import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import onion.loss as loss
from onion import rhn, attention, conv
from vg.simple_data import vector_padder
from vg.scorer import Scorer, testing
from vg.defn.encoders import TextEncoder, SpeechEncoderBottom, SpeechEncoderTop, ImageEncoder
from collections import Counter
import sys
import itertools
import json 

def step(task, *args):

    loss = task.train_cost(*args)
    task.optimizer.zero_grad()
    loss.backward()
    _ = nn.utils.clip_grad_norm(task.parameters(), task.config['max_norm'])
    return loss

class Text(nn.Module):

    def __init__(self, speech_encoder, config):
        super(Text, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        self.TextEncoder = TextEncoder(**config['encoder'])  
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, speech, text):
        speech_enc = self.SpeechEncoderTop(self.SpeechEncoderBottom(speech))
        text_enc = self.TextEncoder(text)
        scores = loss.cosine_matrix(speech_enc, text_enc) 
        cost =  loss.contrastive(scores, margin=self.config['margin_size'])
        return cost            

    def args(self, item):
        return (item['audio'], item['input'].astype('int64'))

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)


class Image(nn.Module):

    def __init__(self, speech_encoder, config):
        super(Image, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        self.ImageEncoder = ImageEncoder(**config['encoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, speech, image):
        speech_enc = self.SpeechEncoderTop(self.SpeechEncoderBottom(speech))
        image_enc = self.ImageEncoder(image)
        scores = loss.cosine_matrix(speech_enc, image_enc) 
        cost =  loss.contrastive(scores, margin=self.config['margin_size'])
        return cost     

    def args(self, item):
        return (item['audio'], item['target_v'])

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)


class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottom(**config['SpeechEncoderBottom'])
        self.Text = Text(self.SpeechEncoderBottom, config['Text'])  
        self.Image = Image(self.SpeechEncoderBottom, config['Image'])
  
    def encode_images(self, images):
        with testing(self):
            rep = self.Image.ImageEncoder(images)
        return rep

    def predict(self, audio):
        with testing(self):
            rep = self.Image.SpeechEncoderTop(self.Image.SpeechEncoderBottom(audio))
        return rep


def experiment(net, data, run_config):
    def valid_loss(name, task):
        result = []
        for item in data[name].iter_valid_batches():
            args = task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append([ x.data.cpu().numpy() for x in task.test_cost(*args) ])
        return result
    
    net.cuda()
    net.train()
    scorer = Scorer(data['Image'].provider, 
                    dict(split='val', 
                         tokenize=lambda x: x['audio'], 
                         batch_size=data['Image'].batch_size))
    last_epoch = 0

    for _, task in run_config['tasks']:
        task.optimizer.zero_grad()

    with open("result.json", "w") as out:
      for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        costs = dict(Text=Counter(), Image=Counter())
        
        for _j, items in enumerate(zip(data['Image'].iter_train_batches(reshuffle=True), 
                                       data['Text'].iter_train_batches(reshuffle=True))):
            j = _j + 1  
            item = dict(Image=items[0], Text=items[1])
            for name, task in run_config['tasks']:
                spk = item[name]['speaker'][0] if len(set(item[name]['speaker'])) == 1 else 'MIXED'
                args = task.args(item[name])
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
              

                loss = task.cost(*args)
                task.optimizer.zero_grad()
                loss.backward()
                _ = nn.utils.clip_grad_norm(task.parameters(), task.config['max_norm'])
                task.optimizer.step()
                costs[name] += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data[name].batch_size, name, spk, "train", "".join([str(costs[name]['cost']/costs[name]['N'])]))

                if j % run_config['validate_period'] == 0:
                   loss = valid_loss(name, task)
                   print(epoch, j, 0, name, "VALID", "valid", "".join([str(numpy.mean(loss))]))


                sys.stdout.flush()
        torch.save(net, "model.{}.pkl".format(epoch))

        with testing(net):
            result = dict(epoch=epoch, rsa=scorer.rsa_image(net), 
                                       retrieval=scorer.retrieval(net), 
                                       speaker_id=scorer.speaker_id(net))
            out.write(json.dumps(result))
            out.write("\n")
            out.flush()


    torch.save(net, "model.pkl")

