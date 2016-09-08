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

tf.app.flags.DEFINE_string("data_dir", "/share/data/speech/Data/ttran/for_batch_jobs/swbd_speech", 
                                       "directory of dictionary files")
tf.app.flags.DEFINE_integer("sp_scale", 5, "scaling of input buckets")
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


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
    words = sentence.strip().split()
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
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, True)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] > len(sent_ids) and _buckets[b][1] > len(parse_ids)]
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


def get_data_stats(data_dir, split, sent_vocab, parse_vocab):
    data_stats = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            mfccs = this_data[k]['mfccs']
            avg_mfccs = summarize_mfcc(mfccs)
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, True)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] > len(sent_ids) and _buckets[b][1] > len(parse_ids)]
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            data_stats[bucket_id].append([len(sent_ids), len(parse_ids), len(avg_mfccs)])
    return data_stats

def print_stats(data_stats):
    for bucket_id in range(len(_buckets)):
        this_bucket = data_stats[bucket_id]
        print("Bucket %d: size = %d" %(bucket_id, len(this_bucket)))
        sent_lengths = [x[0] for x in this_bucket]
        parse_lengths = [x[1] for x in this_bucket]
        mfcc_lengths = [x[2] for x in this_bucket]
        print("Sentence num tokens: min = %d; max = %d; mean = %.2f; median = %.2f" 
                %(min(sent_lengths), max(sent_lengths), np.mean(sent_lengths), np.median(sent_lengths)) )
        print("Parse lengths: min = %d; max = %d; mean = %.2f; median = %.2f" 
                %(min(parse_lengths), max(parse_lengths), np.mean(parse_lengths), np.median(parse_lengths)) )
        print("Num mfcc frames (averaged): min = %d; max = %d; mean = %.2f; median = %.2f" 
                %(min(mfcc_lengths), max(mfcc_lengths), np.mean(mfcc_lengths), np.median(mfcc_lengths)) )
        print("  num sentences where mfccs were cut off (num_avg_frames > bucket_len*spscale): %d" 
            %len([x for x in mfcc_lengths if x > _buckets[bucket_id][0]*FLAGS.sp_scale ]) )
        print("  num sentences where space was a bit wasted (num_avg_frames < half num_frames allowed): %d" 
            %len([x for x in mfcc_lengths if x < _buckets[bucket_id][0]*FLAGS.sp_scale*0.5 ]) )
        print(" ")

    
def main(_):
    data_dir = FLAGS.data_dir
    sent_vocabulary_path = os.path.join(data_dir, 'vocab90000.sents') 
    parse_vocabulary_path = os.path.join(data_dir, 'vocab128.parse')
    parse_vocab, _ = initialize_vocabulary(parse_vocabulary_path)
    sent_vocab, _ = initialize_vocabulary(sent_vocabulary_path)
    
#    train_set = get_data_stats(data_dir, 'train', sent_vocab, parse_vocab)
#    print("train set")
#    print_stats(train_set)
   
#    dev_set = get_data_stats(data_dir, 'dev', sent_vocab, parse_vocab)
#    print("dev set")
#    print_stats(dev_set)

    dev2_set = get_data_stats(data_dir, 'dev2', sent_vocab, parse_vocab)
    print("dev2 set")
    print_stats(dev2_set)

    test_set = get_data_stats(data_dir, 'test', sent_vocab, parse_vocab)
    print("test set")
    print_stats(test_set)

    

if __name__ == "__main__":
    tf.app.run()


