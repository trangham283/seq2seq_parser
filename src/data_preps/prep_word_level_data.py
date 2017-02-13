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
tf.app.flags.DEFINE_string("output_dir", "/s0/ttmt001/speech_parsing", \
        "directory of output files")
tf.app.flags.DEFINE_integer("sp_scale", 10, "scaling of input buckets")
tf.app.flags.DEFINE_integer("avg_frame", 5, "number of frames to average over")
FLAGS = tf.app.flags.FLAGS

data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir

mfcc_dir = data_dir + '/swbd_mfcc'
time_dir = data_dir + '/swbd_parses'
pitch_dir = data_dir + '/swbd_pitch'
pitch_pov_dir = data_dir + '/swbd_pitch_pov'

# timing file columns
# filename \t speaker \t globalID \t start_time \t end_time \n
# cols = ['swfile', 'speaker', 'globalID', 'sent_start', 'sent_end']
hop = 10.0 # in msec

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
    summarized = [np.mean(mfccs[offset:offset+FLAGS.avg_frame, :], axis=0) \
            for offset in start_indices]
    return summarized    

def make_array(frames):
    return np.array(frames).T

# The following functions perform data processing from raw in the steps:
# 1. look up timing to get appropriate mfcc and pitch frames
# 2. convert sentences to token ids
# 3. convert parses to token ids
# 4. put everything in a dictionary
def split_frames(split):
    timing_file = os.path.join(time_dir, split + '.sent_parse.csv')
    timing_df = pandas.read_csv(timing_file, sep='\t')
    words_file = os.path.join(time_dir, split + '.data.csv')
    words_df = pandas.read_csv(words_file, sep='\t')
    words_df.rename(columns={'globalID': 'sent_id'}, inplace=True)
    all_df = pandas.merge(timing_df, words_df, on='sent_id')
    sw_files = set(timing_df.swfile.values)
    for sw in sw_files:
        #print sw
        this_dict = defaultdict(dict) 
        for speaker in ['A', 'B']:
            mfcc_file = os.path.join(mfcc_dir, sw + '-' + speaker + '.pickle')
            pitch_file = os.path.join(pitch_dir, sw + '-' + speaker + '.pickle')
            pitch_pov_file = os.path.join(pitch_pov_dir,sw+'-'+speaker+'.pickle')
            fbank_file = os.path.join(fbank_dir, sw + '-' +speaker+'.pickle')
            try:
                data_mfcc = pickle.load(open(mfcc_file))
            except: 
                print("No mfcc file for ", sw, speaker)
                continue
            try:
                data_pitch = pickle.load(open(pitch_file))
            except: 
                print("No pitch file for ", sw, speaker)
                continue
            try:
                data_pitch_pov = pickle.load(open(pitch_pov_file))
            except: 
                print("No pitch pov file for ", sw, speaker)
            try:
                data_fbank = pickle.load(open(fbank_file))
            except: 
                print("No fbank file for ", sw, speaker)
                continue
            mfccs = data_mfcc.values()[0]
            pitchs = data_pitch.values()[0]
            pitch_povs = data_pitch_pov.values()[0]
            fbanks = data_fbank.values()[0]
            #this_df = all_df[(all_df.swfile==sw)&(all_df.speaker==speaker)]
            # in case both data frames had 'speaker' column:
            this_df = all_df[(all_df.swfile==sw)&(all_df.speaker_x==speaker)]
            for i, row in this_df.iterrows():
                s_ms = row.start*1000 # in msec
                e_ms = row.end*1000


                s_frame = int(np.floor(s_ms / hop))
                e_frame = int(np.ceil(e_ms / hop))
                
                
                
                mf_frames = mfccs[s_frame:e_frame]
                pf_frames = pitchs[s_frame:e_frame]
                pv_frames = pitch_povs[s_frame:e_frame]
                fb_frames = fbanks[s_frame:e_frame]]
                globID = row.globalID.replace('~','_'+speaker+'_')
                this_dict[globID]['sents'] = row.sents
                this_dict[globID]['parse'] = row.parse
                this_dict[globID]['mfccs'] = mf_frames
                this_dict[globID]['pitch2'] = pf_frames
                this_dict[globID]['pitch3'] = pv_frames
                this_dict[globID['fbank'] = fb_frames

        dict_name = os.path.join(output_dir, split, sw + '_word_level.pickle')
        pickle.dump(this_dict, open(dict_name, 'w'))


def process_data_both(data_dir, split, sent_vocab, parse_vocab, acoustic):
    data_set = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            mfccs = make_array(this_data[k]['mfccs'])
            pitch2 = make_array(this_data[k]['pitch2'])
            pitch3 = make_array(this_data[k]['pitch3'])
            fbank = make_array(this_data[k]['fbank'])
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
            if acoustic == 'all':
                data_set[bucket_id].append([sent_ids, parse_ids, mfccs, pitch2, pitch3])
            elif acoustic == 'mfcc':
                data_set[bucket_id].append([sent_ids, parse_ids, mfccs])
            elif acoustic == 'fbank':
                data_set[bucket_id].append([sent_ids, parse_ids, fbank])
            elif acoustic == 'pitch2':
                data_set[bucket_id].append([sent_ids, parse_ids, pitch2])
            else: #acoustic == 'pitch3'
                data_set[bucket_id].append([sent_ids, parse_ids, pitch3])

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
    # processing raw data into dictionary files
    split_frames('dev2')    # choose splits dev/dev2/train/test
    
    #sent_vocabulary_path = os.path.join(data_dir, 'vocab.sents') 
    #parse_vocabulary_path = os.path.join(data_dir, 'vocab.parse')
    #parse_vocab, _ = initialize_vocabulary(parse_vocabulary_path)
    #sent_vocab, _ = initialize_vocabulary(sent_vocabulary_path)
   
    #split = 'dev2'
    #acoustic = 'pitch3'
    #this_set = process_data_both(data_dir, split, sent_vocab, parse_vocab, \
    #        acoustic)
    #this_file = os.path.join(output_dir, split + '_' + acoustic + '.pickle')
    #pickle.dump(this_set, open(this_file,'w'))


if __name__ == "__main__":
    tf.app.run()


