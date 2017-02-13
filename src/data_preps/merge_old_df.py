import gzip
import os
import sys
import random
import glob
import pandas
from collections import defaultdict

import numpy as np
import cPickle as pickle

# timing file columns
# filename \t speaker \t globalID \t start_time \t end_time \n
cols = ['swfile', 'speaker', 'globalID', 'sent_start', 'sent_end']

time_dir = '/s0/ttmt001/speech_parsing/swbd_parses'


def merge_files(split):
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
    file_name = os.path.join(time_dir, split + '.sent_parse.csv')
    all_df.to_csv(file_name, sep='\t', index=False)

if __name__ == "__main__":
    #merge_files('dev2')
    merge_files('dev')
    #merge_files('test')
    #merge_files('train')


