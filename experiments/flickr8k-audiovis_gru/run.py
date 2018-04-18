import numpy
import random
seed = 1235
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.experiment as E
import vg.data_provider as dp
import vg.defn.audiovis_gru as D
dataset = 'flickr8k'
batch_size = 32
epochs=25
prov = dp.getDataProvider(dataset, root='../..', audio_kind='human.max1K.accel3.ord.mfcc')
data = sd.SimpleData(prov, min_df=10, scale=False,
                     batch_size=batch_size, shuffle=True)
model_config = dict(size=1024, depth=4, max_norm=2.0, residual=True,
                    lr=0.0002, size_vocab=37, size_target=4096,
                    filter_length=6, filter_size=64, stride=2,
                    contrastive=True, margin_size=0.2, fixed=True,
                    init_img='xavier', size_attn=128)
run_config = dict(task=D.Visual, epochs=epochs, validate_period=400)



def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=audio, split='val', task=D.Visual, batch_size=batch_size,
                   epochs=epochs, encode_sentences=D.encode_sentences, encode_images=D.encode_images)

E.run_train(data, prov, model_config, run_config, eval_config)
