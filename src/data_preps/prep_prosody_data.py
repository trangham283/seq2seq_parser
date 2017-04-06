import os
import re
import sys
import glob
import pandas
from collections import defaultdict
from tree_utils import merge_sent_tree

import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import cPickle as pickle

tf.app.flags.DEFINE_string("data_dir", "/s0/ttmt001/speech_parsing", \
        "directory of swbd data files")
tf.app.flags.DEFINE_string("output_dir", "/s0/ttmt001/speech_parsing/tmp-prosody", \
        "directory of output files")
tf.app.flags.DEFINE_integer("sp_scale", 10, "scaling of input buckets")
tf.app.flags.DEFINE_integer("avg_frame", 5, "number of frames to average over")
FLAGS = tf.app.flags.FLAGS

data_dir = FLAGS.data_dir
output_dir = FLAGS.output_dir

mfcc_dir = data_dir + '/swbd_mfcc'
time_dir = data_dir + '/swbd_trees'
pitch_dir = data_dir + '/swbd_pitch'
pitch_pov_dir = data_dir + '/swbd_pitch_pov'
fbank_dir = data_dir + '/swbd_fbank'
pause_dir = data_dir + '/pause_data'

hop = 10.0 # in msec
num_sec = 0.04  # amount of time to approximate extra frames when no time info available

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
#_buckets = [(10, 40), (25, 100), (50, 200), (100, 350)]
_buckets = [(10, 55), (25, 110), (50, 200), (100, 350)]

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

def convert_to_array(str_vector):
    str_vec = str_vector.replace('[','').replace(']','').replace(',','').split()
    num_list = []
    for x in str_vec:
        x = x.strip()
        if x != 'None': num_list.append(float(x))
        else: num_list.append(np.nan)
    return num_list

def has_bad_alignment(num_list):
    for i in num_list:
        if i < 0 or np.isnan(i): return True
    return False

def find_bad_alignment(num_list):
    bad_align = []
    for i in range(len(num_list)):
        if num_list[i] < 0 or np.isnan(num_list[i]): 
            bad_align.append(i)
    return bad_align

def has_bad_edge(num_list):
    start = num_list[0]
    end = num_list[-1]   
    if start<-1 or end<-1: return True
    if np.isnan(start) or np.isnan(end): return True
    return False

def prune_punct(num_list, tokens):
    new_list = []
    assert len(num_list)==len(tokens)
    for i in range(len(num_list)):
        if tokens[i] in _PUNCT: continue
        new_list.append(num_list[i])
    return new_list

def check_valid(num):
    if num < 0 or np.isnan(num): return False
    return True

def clean_up_old(stimes, etimes):
    if not check_valid(stimes[-1]):
        stimes[-1] = max(etimes[-1] - num_sec, 0)
    
    if not check_valid(etimes[0]):
        etimes[0] = stimes[0] +  num_sec
    
    for i in range(1,len(stimes)-1):
        this_st = stimes[i]
        prev_st = stimes[i-1]
        next_st = stimes[i+1]

        this_et = etimes[i]
        prev_et = etimes[i-1]
        next_et = etimes[i+1]   
   
        if not check_valid(this_st) and check_valid(prev_et):
            stimes[i] = prev_et

        if not check_valid(this_st) and check_valid(prev_st):
            stimes[i] = prev_st + num_sec

    for i in range(1,len(etimes)-1)[::-1]:
        this_st = stimes[i]
        prev_st = stimes[i-1]
        next_st = stimes[i+1]

        this_et = etimes[i]
        prev_et = etimes[i-1]
        next_et = etimes[i+1]   
        if not check_valid(this_et) and check_valid(next_st):
            etimes[i] = next_st

        if not check_valid(this_et) and check_valid(next_et):
            etimes[i] = next_et - num_sec

    return stimes, etimes

def clean_up(stimes, etimes, tokens, dur_stats):
    total_raw_time = etimes[-1] - stimes[0]
    total_mean_time = sum([dur_stats[w]['mean'] for w in tokens])
    scale = min(total_raw_time / total_mean_time, 1)

    no_start_idx = find_bad_alignment(stimes)
    no_end_idx = find_bad_alignment(etimes)

    # fix start times first
    for idx in no_start_idx:
        if idx not in no_end_idx:
            # this means the word does have an end time; let's use it
            stimes[idx] = etimes[idx] - scale*dur_stats[tokens[idx]]['mean']
        else:
            # this means the idx does not s/e times -- just use prev's start
            stimes[idx] = stimes[idx-1] 

    # now all start times should be there
    for idx in no_end_idx:
        etimes[idx] = stimes[idx] + scale*dur_stats[tokens[idx]]['mean']
            
    return stimes, etimes


def make_dict(pause_data):
    pauses = dict()
    for bucket in pause_data:
        for sample in bucket:
            sent_id, info_dict, parse = sample
            pauses[sent_id] = info_dict
    return pauses


# The following functions perform data processing from raw in the steps:
# 1. look up timing to get appropriate mfcc and pitch frames
# 2. convert sentences to token ids
# 3. convert parses to token ids
# 4. put everything in a dictionary
def split_frames(split, feat_types):
    errfile = os.path.join(output_dir, split + '_frame_long_stats.txt')
    ftoolong = open(errfile, 'w')
    data_file = os.path.join(time_dir, split + '.data.csv')
    pause_file = os.path.join(pause_dir, split+'_nopunc.pickle')
    pause_data = pickle.load(open(pause_file))
    pauses = make_dict(pause_data)
    dur_stats_file = os.path.join(data_dir, 'avg_word_stats.pickle')
    dur_stats = pickle.load(open(dur_stats_file))

    df = pandas.read_csv(data_file, sep='\t')
    sw_files = set(df.file_id.values)
    for sw in sw_files:
        this_dict = defaultdict(dict) 
        for speaker in ['A', 'B']:
            mfcc_file = os.path.join(mfcc_dir, sw + '-' + speaker + '.pickle')
            pitch_file = os.path.join(pitch_dir, sw + '-' + speaker + '.pickle')
            pitch_pov_file = os.path.join(pitch_pov_dir,sw+'-'+speaker+'.pickle')
            fbank_file = os.path.join(fbank_dir, sw + '-' +speaker+'.pickle')

            for feat in feat_types:
                if 'mfcc' in feat_types:
                    try:
                        data_mfcc = pickle.load(open(mfcc_file))
                    except: 
                        print("No mfcc file for ", sw, speaker)
                        continue
                    mfccs = data_mfcc.values()[0]
                if 'pitch2' in feat_types:
                    try:
                        data_pitch = pickle.load(open(pitch_file))
                    except: 
                        print("No pitch file for ", sw, speaker)
                        continue
                    pitchs = data_pitch.values()[0]
                if 'pitch3' in feat_types:
                    try:
                        data_pitch_pov = pickle.load(open(pitch_pov_file))
                    except: 
                        print("No pitch pov file for ", sw, speaker)
                    pitch_povs = data_pitch_pov.values()[0]
                if 'fbank' in feat_types:
                    try:
                        data_fbank = pickle.load(open(fbank_file))
                    except: 
                        print("No fbank file for ", sw, speaker)
                        continue
                    fbanks = data_fbank.values()[0]

            this_df = df[(df.file_id==sw)&(df.speaker==speaker)]
            for i, row in this_df.iterrows():
                tokens = row.sentence.strip().split()
                stimes = convert_to_array(row.start_times)
                etimes = convert_to_array(row.end_times)
                
                if len(stimes)==1: 
                    if (not check_valid(stimes[0])) and (not check_valid(etimes[0])):
                        print "no time available for sentence", row.sent_id
                        continue
                    elif not check_valid(stimes[0]):
                        stimes[0] = max(etimes[0] - dur_stats[tokens[0]]['mean'], 0)
                    else:
                        etimes[0] = stimes[0] + dur_stats[tokens[0]]['mean']
                 
                if check_valid(stimes[0]): 
                    begin = stimes[0]
                else:
                    # cases where the first word is unaligned
                    if check_valid(etimes[0]): 
                        begin = max(etimes[0] - dur_stats[tokens[0]]['mean'], 0) 
                        stimes[0] = begin
                    elif check_valid(stimes[1]):
                        begin = max(stimes[1] - dur_stats[tokens[-1]]['mean'], 0)
                        stimes[0] = begin
                    else:
                        continue

                if check_valid(etimes[-1]): 
                    end = etimes[-1]
                else:
                    # cases where the last word is unaligned
                    if check_valid(stimes[-1]): 
                        end = stimes[-1] + dur_stats[tokens[-1]]['mean']
                        etimes[-1] = end
                    elif check_valid(etimes[-2]):
                        end = etimes[-2] + dur_stats[tokens[-1]]['mean']
                        etimes[-1] = end
                    else:
                        continue
                
                # final clean up
                stimes, etimes = clean_up(stimes, etimes, tokens, dur_stats)
                assert len(stimes) == len(etimes) == len(tokens)

                sframes = [int(np.floor(x*100)) for x in stimes]
                eframes = [int(np.ceil(x*100)) for x in etimes]
                s_frame = sframes[0]
                e_frame = eframes[-1]
                word_lengths = [e-s for s,e in zip(sframes,eframes)]
                invalid = [x for x in word_lengths if x <=0]
                toolong = [x for x in word_lengths if x >=100]
                if len(invalid)>0: 
                    print "End time < start time for: ", row.sent_id, row.speaker 
                    print invalid
                    continue
                if len(toolong)>0:
                    item = row.sent_id + ' ' +row.speaker +' ' + str(toolong) + '\n'
                    ftoolong.write(item)

                offset = s_frame
                word_bounds = [(x-offset,y-offset) for x,y in zip(sframes, eframes)]
                assert len(word_bounds) == len(tokens)
                globID = row.sent_id.replace('~','_'+speaker+'_')
                if globID not in pauses: 
                    print "No pause info for sentence: ", globID
                    continue
                this_dict[globID]['sents'] = row.sentence
                this_dict[globID]['parse'] = row.parse
                this_dict[globID]['windices'] = word_bounds
                this_dict[globID]['word_dur'] = [etimes[i]-stimes[i] for i in range(len(stimes))]
                this_dict[globID]['pause_bef'] = pauses[globID]['pause_bef']
                this_dict[globID]['pause_aft'] = pauses[globID]['pause_aft']
                for feat in feat_types:
                    if feat=='mfcc':
                        mf_frames = mfccs[s_frame:e_frame]
                        this_dict[globID]['mfccs'] = mf_frames
                    if feat=='pitch2':
                        pf_frames = pitchs[s_frame:e_frame]
                        this_dict[globID]['pitch2'] = pf_frames
                    if feat=='pitch3':
                        pv_frames = pitch_povs[s_frame:e_frame]
                        this_dict[globID]['pitch3'] = pv_frames
                    if feat=='fbank':
                        fb_frames = fbanks[s_frame:e_frame]
                        this_dict[globID]['fbank'] = fb_frames

        dict_name = os.path.join(output_dir, split, sw + '_prosody.pickle')
        pickle.dump(this_dict, open(dict_name, 'w'))
    ftoolong.close()


def norm_energy_by_turn(this_data):
    feat_dim = 41
    turnA = np.empty((feat_dim,0)) 
    turnB = np.empty((feat_dim,0))
    for k in this_data.keys():
        fbank = this_data[k]['fbank']
        fbank = np.array(fbank).T
        if 'A' in k:
            turnA = np.hstack([turnA, fbank])
        else:
            turnB = np.hstack([turnB, fbank])
    meanA = np.mean(turnA, 1) 
    stdA = np.std(turnA, 1)
    meanB = np.mean(turnB, 1)
    stdB = np.std(turnB, 1)
    maxA = np.max(turnA, 1)
    maxB = np.max(turnB, 1)
    return meanA, stdA, meanB, stdB, maxA, maxB 

def process_remaining_data(data_dir, split, sent_vocab, parse_vocab):
    data_set = [[] for _ in _buckets]
    sentID_set = [[] for _ in _buckets]
    dur_stats_file = os.path.join(data_dir, 'avg_word_stats.pickle')
    dur_stats = pickle.load(open(dur_stats_file))
    global_mean = np.mean([x['mean'] for x in dur_stats.values()])
    data_file = os.path.join(time_dir, split + '.data.csv')
    pre_data_file = os.path.join(pause_dir, split+'_unproc_nopunc.pickle')
    pre_data = pickle.load(open(pre_data_file))
    df = pandas.read_csv(data_file, sep='\t')
    for bucket_id in range(len(pre_data)):
        for sample in pre_data[bucket_id]:
            if not sample:
                continue
            k, fd, _ = sample
            swfile, speaker, sentnum = k.split('_')
            sent_name_in_df = swfile + '~' + sentnum
            sent_data = df[df.sent_id==sent_name_in_df]
            assert len(sent_data) == 1
            parse = sent_data.parse.values[0]
            sentence = sent_data.sentence.values[0]
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, False, False)
            if split != 'extra':
                parse_ids.append(EOS_ID)
            pause_bef = fd['pause_bef']
            pause_aft = fd['pause_aft'] 
            data_set[bucket_id].append([sent_ids, parse_ids, [], [], \
                    pause_bef, pause_aft, []])
            sentID_set[bucket_id].append(k)

    return data_set, sentID_set


def process_data_both(data_dir, split, sent_vocab, parse_vocab, normalize=False):
    data_set = [[] for _ in _buckets]
    sentID_set = [[] for _ in _buckets]
    dur_stats_file = os.path.join(data_dir, 'avg_word_stats.pickle')
    dur_stats = pickle.load(open(dur_stats_file))
    global_mean = np.mean([x['mean'] for x in dur_stats.values()])
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))

        if normalize:
            meanA, stdA, meanB, stdB, maxA, maxB  = norm_energy_by_turn(this_data)

        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            windices = this_data[k]['windices']
            pause_bef = this_data[k]['pause_bef']
            pause_aft = this_data[k]['pause_aft']

            # features needing normalization
            word_dur = this_data[k]['word_dur']
            pitch3 = make_array(this_data[k]['pitch3'])
            fbank = make_array(this_data[k]['fbank'])
            if normalize:
                exp_fbank = np.exp(fbank)
                # normalize energy by z-scoring
                if 'A' in k:
                    mu = meanA
                    sigma = stdA
                    hi = maxA
                else:
                    mu = meanB
                    sigma = stdB
                    hi = maxB

                #e_total = np.sum(mu[1:])
                #e0 = (fbank[0, :] - mu[0]) / sigma[0]
                #elow = np.sum(fbank[1:21,:],0)/e_total
                #ehigh = np.sum(fbank[21:,:],0)/e_total

                e_total = exp_fbank[0, :]
                e0 = fbank[0, :] / hi[0]
                elow = np.log(np.sum(exp_fbank[1:21,:],0)/e_total)
                ehigh = np.log(np.sum(exp_fbank[21:,:],0)/e_total)

                energy = np.array([e0,elow,ehigh])

                # normalize word durations by dividing by mean
                words = sentence.split()
                assert len(word_dur) == len(words)
                for i in range(len(words)):
                    if words[i] not in dur_stats:
                        print "No mean dur info for word ", words[i]
                        wmean = global_mean
                    wmean = dur_stats[words[i]]['mean']
                    # clip at 5.0
                    word_dur[i] = min(word_dur[i]/wmean, 5.0)
            else:
                energy = fbank[0,:].reshape((1,fbank.shape[1]))

            pitch3_energy = np.vstack([pitch3, energy])

            # convert tokens to ids
            sent_ids = sentence_to_token_ids(sentence, sent_vocab, True, True)
            parse_ids = sentence_to_token_ids(parse, parse_vocab, False, False)
            if split != 'extra':
                parse_ids.append(EOS_ID)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)]
            if not maybe_buckets: 
                print "Sentence does not fit bucket: ", k, len(sent_ids), len(parse_ids)
                continue
            bucket_id = min(maybe_buckets)
            
            data_set[bucket_id].append([sent_ids, parse_ids, windices, pitch3_energy, \
                    pause_bef, pause_aft, word_dur])
            sentID_set[bucket_id].append(k)

    return data_set, sentID_set

def prep_bk_data(data_dir, split):
    treefile = os.path.join(data_dir, split + '_trees_for_bk_new_buckets.mrg')
    ft = open(treefile, 'w')
    sentfile = os.path.join(data_dir, split + '_sents_for_bk_new_buckets.txt')
    fs = open(sentfile, 'w')
    idfile = os.path.join(data_dir, split + '_sentence_ids_flat.txt')
    fi = open(idfile, 'w')
    
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))

        for k in this_data.keys():
            sentence = this_data[k]['sents']
            sent_toks = sentence.strip().split()
            parse = this_data[k]['parse']
            parse_toks = parse.strip().split()

            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_toks) and _buckets[b][1] >= len(parse_toks)+1 ]
            if not maybe_buckets: 
                print "Sentence does not fit bucket: ", k, len(sent_toks), len(parse_toks) + 1
                continue
            bucket_id = min(maybe_buckets)
            fs.write(sentence + '\n')
            merged = merge_sent_tree(parse_toks, sent_toks)
            ft.write(' '.join(merged) + '\n')
            fi.write(k + '\n')
    ft.close()
    fs.close()
    fi.close()
            

def main(_):
    '''
    print "\nCheck dev"
    get_stats('dev')
    print "\nCheck test"
    get_stats('test')
    print "\nCheck train"
    get_stats('train')
    '''
    
    sent_vocabulary_path = os.path.join(output_dir, 'vocab.sents') 
    parse_vocabulary_path = os.path.join(output_dir, 'vocab.parse')
    parse_vocab, _ = initialize_vocabulary(parse_vocabulary_path)
    sent_vocab, _ = initialize_vocabulary(sent_vocabulary_path)

    split = 'train'
 
    # split frames into utterances first
    #feats = ['pitch3', 'fbank'] 
    #split_frames(split, feats)
    #split_frames('test', feats)  # ==> dumps to output_dir
    #split_frames('train', feats)

    # normalize and process data into buckets
    #normalize = True
    #this_set, sentID_set = process_data_both(output_dir, split, sent_vocab, parse_vocab, normalize)
    #this_file = os.path.join(output_dir, split + '_prosody_normed.pickle')
    #pickle.dump(this_set, open(this_file,'w'))
    #sent_file = os.path.join(output_dir, split + '_sentID.pickle')
    #pickle.dump(sentID_set, open(sent_file, 'w'))

    #prep_bk_data(output_dir, 'test')
    
    split = 'test'
    this_set, sentID_set = process_remaining_data(output_dir, split, sent_vocab, parse_vocab)
    this_file = os.path.join(output_dir, split + '_remaining.pickle')
    pickle.dump(this_set, open(this_file,'w'))
    sent_file = os.path.join(output_dir, split + '_remaining_sentID.pickle')
    pickle.dump(sentID_set, open(sent_file, 'w'))

if __name__ == "__main__":
    tf.app.run()


