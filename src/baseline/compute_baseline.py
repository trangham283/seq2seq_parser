#!/usr/bin/env python

# Collect training data, create dictionary for looking up parse output 
# based only on input length in terms of tokens

import os
import sys
import random
import re
import data_utils

import cPickle as pickle
import tensorflow as tf
from tree_utils import add_brackets, match_length, merge_sent_tree
from collections import defaultdict

# also on /share/data/speech/Data/ttran/for_batch_jobs/parses_swbd
data_dir = '/scratch/ttran/Datasets/parses_swbd'
train_sents_file = os.path.join(data_dir, 'train.sents')
train_parse_file = os.path.join(data_dir, 'train.parse')

baseline_dict = defaultdict(list)
sents = open(train_sents_file).readlines()
parses = open(train_parse_file).readlines()
for sent, parse in zip(sents, parses):
    sent_len = len(sent.rstrip().split())
    baseline_dict[sent_len].append(parse.rstrip())

# dev_sents and parse could have been loaded from data_dir too, 
# but I'm loading as in decoding pipeline to ensure consistency
dev_dir = '/share/data/speech/Data/ttran/for_batch_jobs/swbd_speech'
dev_path = os.path.join(dev_dir, 'sw_dev_both.pickle')
dev_set = pickle.load(open(dev_path))

sents_vocab_path = os.path.join(dev_dir,"vocab90000.sents")
parse_vocab_path = os.path.join(dev_dir,"vocab128.parse")
sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
_, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

_buckets = [(10, 40), (25, 85), (40, 150)]

gold_file = 'gold_dev.txt'
baseline_file = 'baseline_dev.txt'
fg = open(gold_file, 'w')
fb = open(baseline_file, 'w')

for bucket_id in xrange(len(_buckets)):
    for sentence in dev_set[bucket_id]:
        slength = len(sentence[0])
        toks = sentence[0]
        gold = sentence[1]
        gold_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in gold]
        sent_text = [tf.compat.as_str(rev_sent_vocab[output]) for output in toks]
        prediction = random.choice(baseline_dict[slength])
        parse_mx = match_length(prediction.split(), sent_text)
        to_write_mx = merge_sent_tree(parse_mx, sent_text)
        to_write_gold = merge_sent_tree(gold_parse, sent_text)
        fg.write('{}\n'.format(' '.join(to_write_gold)))
        fb.write('{}\n'.format(' '.join(to_write_mx)))


fb.close()
fg.close()





