#!/usr/bin/env python

import os
import cPickle as pickle
import pandas

audio_list_file = 'audio_list.txt'
audio_list = open(audio_list_file).readlines()
audio_list = [x.strip() for x in audio_list]

split = 'dev'
pause_dir = '/s0/ttmt001/speech_parsing/pause_data'
tree_dir = '/s0/ttmt001/speech_parsing/swbd_trees'

pause_file = os.path.join(pause_dir, split+'_nopunc.pickle')
tree_file = os.path.join(tree_dir, split + '.data.csv')

def make_dict(pause_data):
    pauses = dict()
    for bucket in pause_data:
        for sample in bucket:
            sent_id, info_dict, parse = sample
            pauses[sent_id] = info_dict
    return pauses

pause_data = pickle.load(open(pause_file))
pauses = make_dict(pause_data)

df = pandas.read_csv(tree_file, sep='\t')
for i, row in df.iterrows():
    mid = '_' + row.speaker + '_'
    sent_name = row.sent_id.replace('~', mid)
    if sent_name in audio_list:
        print sent_name 
        print row.sentence 
        print pauses[sent_name]['pause_bef'] 
        print pauses[sent_name]['pause_aft']
        print row.parse
        print






