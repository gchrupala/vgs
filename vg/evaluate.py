# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
import numpy
from scipy.spatial.distance import cdist

def paraphrase_ranking(vectors, group, ns=(1, 5, 10)):
    """Rank sentences by projection and return evaluation metrics."""
    return ranking(vectors, vectors, group, ns=ns, exclude_self=True)

def ranking(candidates, vectors, correct, ns=(1, 5, 10), exclude_self=False):
    """Rank `candidates` in order of similarity for each vector and return evaluation metrics.

    `correct[i][j]` indicates whether for vector i the candidate j is correct.
    """
    distances = cdist(vectors, candidates, metric='cosine')
    #distances = Cdist(batch_size=2**13)(vectors, candidates)
    result = {'ranks' : [] , 'precision' : {}, 'recall' : {}, 'overlap' : {} }
    for n in ns:
        result['precision'][n] = []
        result['recall'][n]    = []
        result['overlap'][n]   = []
    for j, row in enumerate(distances):
        ranked = numpy.argsort(row)
        if exclude_self:
            ranked = ranked[ranked!=j]
        id_correct = numpy.where(correct[j][ranked])[0]
        rank1 = id_correct[0] + 1
        topn = {}
        for n in ns:
            id_topn = ranked[:n]
            overlap = len(set(id_topn).intersection(set(ranked[id_correct])))
            result['precision'][n].append(overlap/n)
            result['recall'   ][n].append(overlap/len(id_correct))
            result['overlap'  ][n].append(overlap)
        result['ranks'].append(rank1)
    return result
