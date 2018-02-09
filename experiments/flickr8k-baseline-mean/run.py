import numpy
import vg.simple_data as sd
import vg.experiment as E
import vg.data_provider as dp
import vg.defn.baseline_mfcc as D
dataset = 'flickr8k'

prov = dp.getDataProvider(dataset, root='../..', audio_kind='human.max1K.accel3.ord.mfcc')

def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=audio, split='val', encode_sentences=D.aggregate_mean, para=True)


numpy.save("scores.1.npy", D.eval_para(prov, eval_config))


