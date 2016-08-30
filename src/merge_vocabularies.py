#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np

data_dir = '/scratch/ttran/Datasets'

bktotal_sent_vocab = open(os.path.join(data_dir, 'bktotal_data', 'vocab90000.sents'), 'r').readlines()
bktotal_parse_vocab = open(os.path.join(data_dir, 'bktotal_data', 'vocab128.parse'), 'r').readlines() 

swbd_sent_vocab = open(os.path.join(data_dir, 'parses_swbd', 'temp.swbd.vocab.sents'), 'r').readlines() 
swbd_parse_vocab = open(os.path.join(data_dir, 'parses_swbd', 'temp.swbd.vocab.parse'), 'r').readlines() 

bktotal_sent_vocab = [x.rstrip() for x in bktotal_sent_vocab]
bktotal_parse_vocab = [x.rstrip() for x in bktotal_parse_vocab]

swbd_sent_vocab = [x.rstrip() for x in swbd_sent_vocab]
swbd_parse_vocab = [x.rstrip() for x in swbd_parse_vocab]

for i, w in enumerate(swbd_sent_vocab):
    if w[-1] == '-':
        swbd_sent_vocab[i] = '_UNF'  # unfinished word normalization

sentset_swbd = set(swbd_sent_vocab)
parseset_swbd = set(swbd_parse_vocab)

for w in bktotal_sent_vocab:
    if len(sentset_swbd) >= 90000: break
    if w not in sentset_swbd: sentset_swbd.add(w)

for w in bktotal_parse_vocab:
    if len(parseset_swbd) >= 128: break
    if w not in parseset_swbd: parseset_swbd.add(w)

_START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK', '_UNF']

v90k = _START_VOCAB + sorted(list(sentset_swbd.difference(set(_START_VOCAB))))
v128 = _START_VOCAB + sorted(list(parseset_swbd.difference(set(_START_VOCAB))))

fsent = open(os.path.join(data_dir, 'parses_swbd', 'vocab90000.sents'),'w') 
for w in v90k:
    fsent.write('{}\n'.format(w))

fparse = open(os.path.join(data_dir, 'parses_swbd', 'vocab128.parse'),'w') 
for w in v128:
    fparse.write('{}\n'.format(w))

