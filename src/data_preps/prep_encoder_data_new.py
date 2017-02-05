# Use data_utils and seq2seq model specific in this directory
#   this file only uses the read_data portion of the code to
#   preprocess data into buckets
# Additionally, include MFCC info and scale them down to 
#   managable size

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import math
import os
import re
import sys
import tarfile
import random
import glob

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.platform import gfile
import tensorflow as tf
import cPickle as pickle

tf.app.flags.DEFINE_string("data_dir", "/scratch/ttran/Datasets/swbd_speech", "directory of dictionary files")
#tf.app.flags.DEFINE_string("output_dir", "/scratch/ttran/Datasets/swbd_speech", "directory of output files")
tf.app.flags.DEFINE_string("output_dir", "/scratch/ttran/", "directory of output files")
tf.app.flags.DEFINE_integer("sp_scale", 10, "scaling of input buckets")
tf.app.flags.DEFINE_integer("avg_frame", 5, "number of frames to average over")
FLAGS = tf.app.flags.FLAGS

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_UNF = b"_UNF"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _UNF]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
UNF_ID = 4

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      lower=True, tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.
  """
  
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        if tokenizer == 'nltk':
          tokens = nltk.word_tokenize(line)
        elif tokenizer == 'basic':
          tokens = basic_tokenizer(line)
        else:
          tokens = line.strip().split()
        for w in tokens:
        # add lower casing
          if lower: w = w.lower()  
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          word = _UNF if w[-1]=='-' else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True, lower=False):
    words = sentence.strip().split()
    if lower: words = [w.lower() for w in words]
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize UNF and digits by 0 before looking words up in the vocabulary.
    for i, w in enumerate(words):
        if w[-1] == '-': words[i] = _UNF  # unfinished word normalization
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def summarize_mfcc(mfccs):
    start_indices = np.arange(0, len(mfccs), FLAGS.avg_frame)
    mfccs = np.array(mfccs)
    summarized = [np.mean(mfccs[offset:offset+FLAGS.avg_frame, :], axis=0) for offset in start_indices]
    return summarized    

def write_text(data_dir, split, sent_text_path, parse_text_path):
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    fsent = open(sent_text_path, 'w')
    fparse = open(parse_text_path, 'w')
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            fsent.write('{}\n'.format(sentence))
            fparse.write('{}\n'.format(parse))
    fsent.close()
    fparse.close()

def process_data(data_dir, split, sent_vocab, parse_vocab):
    data_set = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            mfccs = this_data[k]['mfccs']
            avg_mfccs = summarize_mfcc(mfccs)
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, False, False)
            if split != 'extra':
                parse_ids.append(EOS_ID)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)]
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            max_frame_len = _buckets[bucket_id][0] * FLAGS.sp_scale
            padded_mfccs = np.zeros( (len(mfccs[0]), max_frame_len) )
            for frame in range(min(max_frame_len, len(avg_mfccs) )):
                padded_mfccs[:, frame] = avg_mfccs[frame]
            data_set[bucket_id].append([sent_ids, parse_ids, padded_mfccs])
    return data_set

def process_data_nopad(data_dir, split, sent_vocab, parse_vocab):
    data_set = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            mfccs = this_data[k]['mfccs']
            avg_mfccs = summarize_mfcc(mfccs)
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, False, False)
            if split != 'extra':
                parse_ids.append(EOS_ID)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)]
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            data_set[bucket_id].append([sent_ids, parse_ids, avg_mfccs])
    return data_set

def map_sentences(data_dir, split):
    mappings = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            sent_ids = sentence.rstrip().split() 
            parse_ids = parse.rstrip().split()
            #include line below for swbd_new, but not for swbd_speech and swbd_tune
            if split != 'extra':
                parse_ids.append(EOS_ID)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)]
                #if _buckets[b][0] > len(sent_ids) and _buckets[b][1] > len(parse_ids)]
            # > for swbd_speech; >= for swbd_new
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            mappings[bucket_id].append(k)
    return mappings 

    
def main(_):
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
#    text_path = '/tmp/swbd_text.txt'
#    parse_path = '/tmp/swbd_parse.txt'
#    max_sent_vocab = 90000
#    max_parse_vocab = 128
#    sent_vocabulary_path = os.path.join(output_dir, 'vocab.sents') 
#    parse_vocabulary_path = os.path.join(output_dir, 'vocab.parse')
    
    # write plain text files
#    write_text(data_dir, 'train', text_path, parse_path)

    # create vocabularies
#    create_vocabulary(sent_vocabulary_path, text_path, max_sent_vocab,
#                     lower=True, tokenizer=None, normalize_digits=True)
#    create_vocabulary(parse_vocabulary_path, parse_path, max_parse_vocab,
#                     lower=False, tokenizer=None, normalize_digits=True)

#    parse_vocab, _ = initialize_vocabulary(parse_vocabulary_path)
#    sent_vocab, _ = initialize_vocabulary(sent_vocabulary_path)
#   
#    print("Processing Train set")
#    train_set = process_data_nopad(data_dir, 'train', sent_vocab, parse_vocab)
#    train_file = os.path.join(output_dir, 'sw_train_both.pickle')
#    pickle.dump(train_set, open(train_file,'w'))
#    
#    print("Processing Dev set")
#    dev_set = process_data_nopad(data_dir, 'dev', sent_vocab, parse_vocab)
#    dev_file = os.path.join(output_dir, 'sw_dev_both.pickle')
#    pickle.dump(dev_set, open(dev_file,'w'))
#
#    print("Processing Dev2 set")
#    dev2_set = process_data_nopad(data_dir, 'dev2', sent_vocab, parse_vocab)
#    dev2_file = os.path.join(output_dir, 'sw_dev2_both.pickle')
#    pickle.dump(dev2_set, open(dev2_file,'w'))
#
#    print("Processing Test set")
#    test_set = process_data_nopad(data_dir, 'test', sent_vocab, parse_vocab)
#    test_file = os.path.join(output_dir, 'sw_test_both.pickle')
#    pickle.dump(test_set, open(test_file,'w'))
#
#    print("Processing Extra set")
#    extra_set = process_data_nopad(data_dir, 'extra', sent_vocab, parse_vocab)
#    extra_file = os.path.join(output_dir, 'sw_extra_both.pickle')
#    pickle.dump(extra_set, open(extra_file,'w'))

    print("Processing Dev set")
    dev_maps = map_sentences(data_dir, 'dev')
    dev_file = os.path.join(output_dir, 'mappings_dev.pickle')
    pickle.dump(dev_maps, open(dev_file,'w'))
    

if __name__ == "__main__":
    tf.app.run()


