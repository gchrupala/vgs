#!/usr/bin/env python3
import numpy
import sys
import os
import logging
import argparse

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    retrieve_p = commands.add_parser('retrieve')
    retrieve_p.set_defaults(func=retrieve)
    retrieve_p.add_argument('dir', nargs='?', help='Target directory', default=".")

    correlate_p = commands.add_parser('correlate')
    correlate_p.add_argument('dir', nargs='?', help='Target directory', default=".")
    correlate_p.set_defaults(func=correlate)
    args = parser.parse_args()
    args.func(args)    


def correlate(args):
    os.chdir(args.dir)

    print("epoch r")
    for i in range(1,26):
       try:
          data = numpy.load("scores.{}.npy".format(i))
          print(i, round(data[0], 3))
       except IOError:
          pass
    
def retrieve(args):
    os.chdir(args.dir)
    
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


if __name__ == '__main__':
    main()
