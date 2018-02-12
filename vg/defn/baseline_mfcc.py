import numpy
from vg.evaluate import paraphrase_ranking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

def eval_corr(prov, config):
        sents = list(prov.iterSentences(split=config['split']))
        split_text = [ s['raw'] for s in sents ]
        vec = TfidfVectorizer(analyzer='char', ngram_range=(2,7))
        split_vecs = vec.fit_transform(split_text)        
        SIM_tfidf = cosine_similarity(split_vecs) 
        sents_tok =  [ config['tokenize'](sent) for sent in sents ]
        pred = config['encode_sentences'](sents_tok)
        SIM_pred = cosine_similarity(pred)
        result = scipy.stats.spearmanr(SIM_tfidf.flatten(), SIM_pred.flatten()) 
        return result

def eval_para(prov, config):
        sents = list(prov.iterSentences(split=config['split']))
        sents_tok =  [ config['tokenize'](sent) for sent in sents ]
        reps = config['encode_sentences'](sents_tok)
        correct_para = numpy.array([ [ sents[i]['imgid']==sents[j]['imgid']
                                      for j in range(len(sents)) ]
                                    for i in range(len(sents)) ] )
        result = paraphrase_ranking(reps, correct_para)
        return result

def aggregate_mean(audios): 
    # audios: list of arrays
    return numpy.vstack([ audio.mean(axis=0) for audio in audios ])

def aggregate_ss(audios):
    return numpy.vstack([ numpy.hstack([audio.mean(axis=0),
                                        audio.min(axis=0),
                                        audio.max(axis=0),
                                        audio.std(axis=0) ]) for audio in audios ])

def aggregate_random(audios):
    return numpy.vstack([ numpy.random.normal(1.0, 0.0, (13,)) for audio in audios ])

