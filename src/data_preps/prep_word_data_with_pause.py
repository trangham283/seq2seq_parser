import os
import re
import sys
import glob
import pandas
from collections import defaultdict

import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import cPickle as pickle

tf.app.flags.DEFINE_string("data_dir", "/s0/ttmt001/speech_parsing", \
        "directory of swbd data files")
tf.app.flags.DEFINE_string("output_dir", "/s0/ttmt001/speech_parsing/word_level", \
        "directory of output files")
FLAGS = tf.app.flags.FLAGS

data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir

mfcc_dir = data_dir + '/swbd_mfcc'
time_dir = data_dir + '/swbd_trees'
pitch_dir = data_dir + '/swbd_pitch'
pitch_pov_dir = data_dir + '/swbd_pitch_pov'
fbank_dir = data_dir + '/swbd_fbank'
pause_dir = data_dir + '/pause_data'

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
_PUNCT = ["'", "`",'"', ",", ".", "/", "?", "[", "]", "(", ")", "{", "}", ":",
";", "!"]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
UNF_ID = 4

# Use the following buckets: 
#_buckets = [(10, 40), (25, 85), (40, 150)]
_buckets = [(10, 40), (25, 100), (50, 200), (100, 350)]

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

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

def make_array(frames):
    return np.array(frames).T

def make_dict(pause_data):
    pauses = dict()
    for bucket in pause_data:
        for sample in bucket:
            sent_id, info_dict, parse = sample
            pauses[sent_id] = info_dict
    return pauses

def process_data_both(data_dir, split, sent_vocab, parse_vocab, normalize=False):
    pause_file = os.path.join(pause_dir, split+'_nopunc.pickle')
    pause_data = pickle.load(open(pause_file))
    pauses = make_dict(pause_data)
    data_set = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            if k not in pauses:
                print "No pause info for sentence ", k
                continue
            pause_bef = pauses[k]['pause_bef']
            pause_aft = pauses[k]['pause_aft']
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            windices = this_data[k]['windices']
            pitch3 = make_array(this_data[k]['pitch3'])
            fbank = make_array(this_data[k]['fbank'])
            if normalize:
                dname = os.path.join(data_dir, 'fbank_mean.pickle')
                mean_vec = pickle.load(open(dname))
                fbank = fbank - mean_vec.reshape((mean_vec.shape[0],1))
                dname = os.path.join(data_dir, 'fbank_var.pickle')
                var_vec = pickle.load(open(dname))
                fbank = fbank / np.sqrt(var_vec.reshape(var_vec.shape[0],1)) 
            energy = fbank[0,:].reshape((1,fbank.shape[1]))
            pitch3_energy = np.vstack([pitch3, energy])
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
            data_set[bucket_id].append([sent_ids, parse_ids, windices, pitch3_energy, \
                    pause_bef, pause_aft])
    return data_set

def main(_):
    sent_vocabulary_path = os.path.join(data_dir, 'vocab.sents') 
    parse_vocabulary_path = os.path.join(data_dir, 'vocab.parse')
    parse_vocab, _ = initialize_vocabulary(parse_vocabulary_path)
    sent_vocab, _ = initialize_vocabulary(sent_vocabulary_path)

    split = 'test'
    # process data into buckets
    normalize = True
    this_set = process_data_both(output_dir, split, sent_vocab, parse_vocab, normalize)
    this_file = os.path.join(output_dir, split + '_p3f4norm_pause.pickle')
    pickle.dump(this_set, open(this_file,'w'))

if __name__ == "__main__":
    tf.app.run()


