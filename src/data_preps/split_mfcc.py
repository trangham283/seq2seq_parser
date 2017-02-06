#!/user/bin/env python

import os
import sys
import pandas
import numpy as np
import cPickle as pickle
from collections import defaultdict

mfcc_dir = '/scratch/ttran/Datasets/swbd_mfcc'
time_dir = '/scratch/ttran/Datasets/parses_swbd'
out_dir = '/scratch/ttran/Datasets/swbd_speech'

# timing file columns
# filename \t speaker \t globalID \t start_time \t end_time \n
cols = ['swfile', 'speaker', 'globalID', 'start', 'end']
hop = 10.0 # in msec


#for split in ['train', 'dev', 'dev2', 'test']:
for split in ['dev']:
    timing_file = os.path.join(time_dir, split+'.times')
    text_file = os.path.join(time_dir, split+'.sents')
    parse_file = os.path.join(time_dir, split+'.parse')
    timing_df = pandas.read_csv(timing_file, sep='\t', names=cols)
    text_df = pandas.read_csv(text_file, sep = '\t', names=['sents'])
    parse_df = pandas.read_csv(parse_file, sep = '\t', names=['parse'])
    timing_df['row_id']=range(len(timing_df))
    text_df['row_id']=range(len(timing_df))
    parse_df['row_id']=range(len(timing_df))
    tree_df = pandas.merge(text_df, parse_df, on='row_id')
    all_df = pandas.merge(timing_df, tree_df, on='row_id')
    sw_files = set(timing_df.swfile.values)
    for sw in sw_files:
        print sw
        this_dict = defaultdict(dict) 
        for speaker in ['A', 'B']:
            mfcc_file = os.path.join(mfcc_dir, sw + '-' + speaker + '.pickle')
            try:
                data = pickle.load(open(mfcc_file))
            except: 
                print "No mfcc file for ", sw, speaker
                continue
            mfccs = data.values()[0]
            this_df = all_df[(all_df.swfile==sw)&(all_df.speaker==speaker)]
            for i, row in this_df.iterrows():
                s_ms = row.start*1000 # in msec
                e_ms = row.end*1000
                s_frame = int(np.floor(s_ms / hop))
                e_frame = int(np.ceil(e_ms / hop))
                mf_frames = mfccs[s_frame:e_frame]
                globID = row.globalID.replace('~','_'+speaker+'_')
                this_dict[globID]['sents'] = row.sents
                this_dict[globID]['parse'] = row.parse
                this_dict[globID]['mfccs'] = mf_frames
        dict_name = os.path.join(out_dir, split, sw + '_data.pickle')
        #pickle.dump(this_dict, open(dict_name, 'w'))

