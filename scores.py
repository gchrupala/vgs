#!/usr/bin/env python3
import numpy
import sys
import os

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

print("epoch R@1 R@5 R@10 rank")
for i in range(1,26):
    try:
        data = numpy.load("scores.{}.npy".format(i)).item(0)
        recall = data['recall']
        print (i, numpy.mean(recall[1]),\
                  numpy.mean(recall[5]),\
                  numpy.mean(recall[10]),\
                  numpy.median(data['ranks']))
    except IOError:
        pass
