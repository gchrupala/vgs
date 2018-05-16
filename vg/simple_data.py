import numpy
import pickle
import gzip
import os
import copy
from onion.util import autoassign
import onion.util as util
from  sklearn.preprocessing import StandardScaler, LabelEncoder
import string
import random
import itertools

# Types of tokenization

def words(sentence):
    return sentence['tokens']

def characters(sentence):
    return list(sentence['raw'])

def compressed(sentence):
    return [ c.lower() for c in sentence['raw'] if c in string.letters ]

def phonemes(sentence):
    return [ pho for pho in sentence['ipa'] if pho != "*" ]

class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class InputScaler():

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        flat = numpy.vstack(data)
        self.scaler.fit(flat)
        return [ self.scaler.transform(X) for X in data ]

    def transform(self, data):
        return [ self.scaler.transform(X) for X in data ]

    def inverse_transform(self, data):
        return [ self.scaler.inverse_transform(X) for X in data ]

def vector_padder(vecs):
        """Pads each vector in vecs with zeros at the beginning. Returns 3D tensor with dimensions:
           (BATCH_SIZE, SAMPLE_LENGTH, NUMBER_FEATURES).
        """

        max_len = max(map(len, vecs))
        #for vec in vecs:
        #    assert len(vec.shape) == 2, "Broken vector {}".format(vec)
                
        return numpy.array([ numpy.vstack([numpy.zeros((max_len-len(vec), vec.shape[1])) , vec])
                            for vec in vecs ], dtype='float32')

class Batcher(object):

    def __init__(self, mapper, pad_end=False, visual=True, erasure=(5,5), sigma=None, noise_tied=False, midpoint=False):
        autoassign(locals())
        self.BEG = self.mapper.BEG_ID
        self.END = self.mapper.END_ID
        try:
            self.gap_low = self.erasure[0]
            self.gap_high = self.erasure[1]
        except:
            self.gap_low = self.erasure
            self.gap_high = self.erasure + 1
        
    def pad(self, xss): # PAD AT BEGINNING
        max_len = max((len(xs) for xs in xss))
        def pad_one(xs):
            if self.pad_end:
                return xs + [ self.END for _ in range(0,(max_len-len(xs))) ]
            return [ self.BEG for _ in range(0,(max_len-len(xs))) ] + xs
        return [ pad_one(xs) for xs in xss ]

    def batch_inp(self, sents):
        mb = self.padder(sents)
        return mb[:,1:]

    def padder(self, sents):
        return numpy.array(self.pad([[self.BEG]+sent+[self.END] for sent in sents]), dtype='int32')


    def batch(self, gr):
        """Prepare minibatch.
        Returns:
        - input string
        - visual target vector
        - output string at t-1
        - target string
        """
        L_tok =  [len(x['tokens_in']) for x in gr]
        L_aud =  [len(x['audio']) for x in gr ]
        mb_inp = self.padder([x['tokens_in'] for x in gr])
        mb_target_t = self.padder([x['tokens_out'] for x in gr])
        inp = mb_inp[:,1:]
        target_t = mb_target_t[:,1:]
        target_prev_t = mb_target_t[:,0:-1]
        target_v = numpy.array([ x['img'] for x in gr ], dtype='float32')
        audio = vector_padder([ x['audio'] for x in gr ]) if gr[0]['audio']  is not None else None
        if self.midpoint:
            mid = midpoint(audio.shape[1] , max(L_aud) - int(numpy.median(L_aud)))
        else:
            mid = audio.shape[1] // 2
        gap = numpy.random.randint(self.gap_low, self.gap_high, 1)[0]
        audio_beg = audio[:, :mid - gap, :]
        audio_end = audio[:, mid + gap:, :]
        if self.midpoint:
            mid = midpoint(inp.shape[1], max(L_tok) - int(numpy.median(L_tok)))
        else:
            mid = inp.shape[1] // 2
        if gap >= mid: # avoid empty arrays
            inp_beg = inp[:, :mid]
            inp_end = inp[:, mid:]
        else:
            inp_beg = inp[:, :mid - gap]
            inp_end = inp[:, mid + gap:]
        
        if self.sigma is not None and not self.noise_tied:
            if numpy.random.binomial(1, 0.5) == 1:
                audio_beg += numpy.random.normal(loc=0.0, scale=self.sigma, size=audio_beg.shape)
            else:
                audio_end += numpy.random.normal(loc=0.0, scale=self.sigma, size=audio_end.shape)
        # Time tied noise
        elif self.sigma is not None and self.noise_tied:
            if numpy.random.binomial(1, 0.5) == 1:
                audio_beg += numpy.random.normal(loc=0.0, scale=self.sigma, size=(audio_beg.shape[0], 1, audio_beg.shape[2]))
            else:
                audio_end += numpy.random.normal(loc=0.0, scale=self.sigma, size=(audio_end.shape[0], 1, audio_end.shape[2]))

        one3 = audio.shape[1] // 3
        two3 = one3 * 2
        audio_1      = audio[:, 1:one3,     :]
        audio_1_prev = audio[:, 0:one3-1,   :]
        audio_2      = audio[:, one3:two3,  :] 
        audio_3      = audio[:, two3+1:,    :]
        audio_3_prev = audio[:, two3:-1,    :]

        assert audio_1.shape == audio_1_prev.shape
        assert audio_3.shape == audio_3_prev.shape
      
        return { 'input': inp,
                 'input_beg': inp_beg,
                 'input_end': inp_end,
                 'target_v':target_v if self.visual else None,
                 'target_prev_t':target_prev_t,
                 'target_t':target_t,
                 'audio': audio,
                 'audio_beg': audio_beg,
                 'audio_end': audio_end,
                 'audio_1': audio_1,
				 'audio_1_prev': audio_1_prev,
                 'audio_2': audio_2,
                 'audio_3_prev': audio_3_prev,
                 'audio_3': audio_3,
                 'speaker': numpy.array([ x['speaker'] for x in gr ]),
                 'speaker_id': numpy.array([ x['speaker_id'] for x in gr ])
                }

def midpoint(L_tot, L_pad):
    return (L_tot - L_pad) // 2 + L_pad

def scale_utterance(data):
    def scale(datum):
        # time x feature
        mu = datum.mean(axis=1, keepdims=True)
        sigma = datum.std(axis=1, keepdims=True)
        return (datum - mu)/sigma
    return [ scale(datum) for datum in data ]

class SimpleData(object):
    """Training / validation data prepared to feed to the model."""
    def __init__(self, provider, tokenize=words, min_df=10, scale=True, scale_input=False, scale_utt=False,
                batch_size=64, shuffle=False, limit=None, curriculum=False, by_speaker=False, val_vocab=False,
                visual=True, erasure=5, midpoint=False, sigma=None, noise_tied=False, speakers=None):
        autoassign(locals())
        self.data = {}
        self.mapper = IdMapper(min_df=self.min_df)
        self.scaler = StandardScaler() if scale else NoScaler()
        self.audio_scaler = InputScaler() if scale_input else NoScaler()
        self.speaker_encoder = LabelEncoder()
        parts = insideout(self.shuffled(arrange(provider.iterImages(split='train'),
                                                               tokenize=self.tokenize,
                                                               limit=limit,
                                                               speakers=speakers )))
        parts_val = insideout(self.shuffled(arrange(provider.iterImages(split='val'), tokenize=self.tokenize)))
        # TRAINING
        if self.val_vocab:
            _ = list(self.mapper.fit_transform(parts['tokens_in'] + parts_val['tokens_in']))
            parts['tokens_in'] = self.mapper.transform(parts['tokens_in']) # FIXME UGLY HACK
        else:
            parts['tokens_in'] = self.mapper.fit_transform(parts['tokens_in'])

        parts['tokens_out'] = self.mapper.transform(parts['tokens_out'])
        parts['img'] = self.scaler.fit_transform(parts['img'])
        self.speaker_encoder.fit(parts['speaker']+parts_val['speaker'])
        parts['speaker_id'] = self.speaker_encoder.transform(parts['speaker'])
        if scale_input:
            parts['audio'] = self.audio_scaler.fit_transform(parts['audio'])
        elif scale_utt:
            parts['audio'] = scale_utterance(parts['audio'])
            
        self.data['train'] = outsidein(parts)

        # VALIDATION
        parts_val['tokens_in'] = self.mapper.transform(parts_val['tokens_in'])
        parts_val['tokens_out'] = self.mapper.transform(parts_val['tokens_out'])
        if self.visual:
            parts_val['img'] = self.scaler.transform(parts_val['img'])
        if scale_input:
            
            parts_val['audio'] = self.audio_scaler.transform(parts_val['audio'])
        elif scale_utt:
            parts_val['audio'] = scale_utterance(parts_val['audio'])
        parts_val['speaker_id'] = self.speaker_encoder.transform(parts_val['speaker'])
        self.data['valid'] = outsidein(parts_val)
        self.batcher = Batcher(self.mapper, pad_end=False, visual=visual, erasure=erasure, sigma=sigma, noise_tied=noise_tied, midpoint=midpoint)

    def shuffled(self, xs):
        if not self.shuffle:
            return xs
        else:
            zs = copy.copy(list(xs))
            random.shuffle(zs)
            return zs


    def iter_train_batches(self, reshuffle=False):
        # sort data by length
        if self.curriculum:
            data = [self.data['train'][i] for i in numpy.argsort([len(x['tokens_in']) for x in self.data['train']])]
        else:
            data = self.data['train']
        if self.by_speaker:
            for x in randomized(by_speaker(self.batcher, data)):
                yield x
        else:                    
            if reshuffle:
                data = randomized(self.data['train'])
            for bunch in util.grouper(data, self.batch_size*20):
                bunch_sort = [ bunch[i] for i in numpy.argsort([len(x['tokens_in']) for x in bunch]) ]
                for item in util.grouper(bunch_sort, self.batch_size):
                    yield self.batcher.batch(item)

    def iter_valid_batches(self):
        if self.by_speaker:
            for x in by_speaker(self.batcher, self.data['valid']):
                yield x
        else:
            for bunch in util.grouper(self.data['valid'], self.batch_size*20):
                bunch_sort = [ bunch[i] for i in numpy.argsort([len(x['tokens_in']) for x in bunch]) ]
                for item in util.grouper(bunch_sort, self.batch_size):
                    yield self.batcher.batch(item)


    def dump(self, model_path):
        """Write scaler and batcher to disc."""
        pickle.dump(self.scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.batcher, gzip.open(os.path.join(model_path, 'batcher.pkl.gz'), 'w'),
                    protocol=pickle.HIGHEST_PROTOCOL)

def by_speaker(batcher, data, batch_size=32):
      speaker = lambda x: x['speaker']
      for _, bunch in itertools.groupby(sorted(data, key=speaker), speaker):
          for item in util.grouper(bunch, batch_size):
             yield batcher.batch(item)

def randomized(data):
    return sorted(data, key= lambda _: random.random())

def arrange(data, tokenize=words, limit=None, speakers=None):
    for i,image in enumerate(data):
        if limit is not None and i > limit:
            break
        for sent in image['sentences']:
            speaker = sent.get('speaker')
            if speakers is None or speaker in speakers:
                toks = tokenize(sent)
                yield {'tokens_in':  toks,
                       'tokens_out': toks,
                       'audio':       sent.get('audio'),
                       'img':        image.get('feat'),
                       'speaker':   speaker
                        }


def insideout(ds):
    """Transform a list of dictionaries to a dictionary of lists."""
    ds  = list(ds)
    result = dict([(k, []) for k in ds[0].keys()])
    for d in ds:
        for k,v in d.items():
            result[k].append(v)
    return result

def outsidein(d):
    """Transform a dictionary of lists to a list of dictionaries."""
    ds = []
    keys = d.keys()
    for key in keys:
        d[key] = list(d[key])
    for i in  range(len(list(d.values())[0])):
        ds.append(dict([(k, d[k][i]) for k in keys]))
    return ds

class IdTable(object):
    """Map hashable objects to ints and vice versa."""
    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.max = 0

    def to_id(self, s, default=None):
        i = self.encoder.get(s, default)
        if i is not None:
            return i
        else:
            i = self.max
            self.encoder[s] = i
            self.decoder[i] = s
            self.max += 1
            return i

    def from_id(self, i):
        return self.decoder[i]


class IdMapper(object):
    """Map lists of words to lists of ints."""
    def __init__(self, min_df=1):
        self.min_df = min_df
        self.freq = {}
        self.ids = IdTable()
        self.BEG = '<BEG>'
        self.END = '<END>'
        self.UNK = '<UNK>'
        self.BEG_ID = self.ids.to_id(self.BEG)
        self.END_ID = self.ids.to_id(self.END)
        self.UNK_ID = self.ids.to_id(self.UNK)

    def size(self):
        return len(self.ids.encoder)

    def fit(self, sents):
        """Prepare model by collecting counts from data."""
        sents = list(sents)
        for sent in sents:
            for word in set(sent):
                self.freq[word] = self.freq.get(word, 0) + 1

    #FIXME .fit(); .transform() should have the same effect as .fit_transform()

    def fit_transform(self, sents):
        """Map each word in sents to a unique int, adding new words."""
        sents = list(sents)
        self.fit(sents)
        return self._transform(sents, update=True)

    def transform(self, sents):
        """Map each word in sents to a unique int, without adding new words."""
        return self._transform(sents, update=False)

    def _transform(self, sents, update=False):
        default = None if update else self.UNK_ID
        for sent in sents:
            ids = []
            for word in sent:
                if self.freq.get(word, 0) < self.min_df:
                    ids.append(self.UNK_ID)
                else:
                    ids.append(self.ids.to_id(word, default=default))
            yield ids

    def inverse_transform(self, sents):
        """Map each id in sents to the corresponding word."""
        for sent in sents:
            yield [ self.ids.from_id(i) for i in sent ]
