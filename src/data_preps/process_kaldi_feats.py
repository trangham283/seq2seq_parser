#!/user/bin/env python

import os
import sys
import pandas
import argparse
import numpy as np
import cPickle as pickle
import glob

#feattype = 'pitch'
#numc = 2 # number of feature dimensions 
feattype = 'fbank'
numc = 41 # number of feature dimensions 

nsplit = 4 # number of splits when kaldi was called

raw_dir = '/s0/ttmt001/speech_parsing/'+feattype
output_dir = '/s0/ttmt001/speech_parsing/swbd_'+feattype+'/'

done_files = glob.glob(output_dir + '*')
done_files = [os.path.basename(x).split('.')[0] for x in done_files]

for i in range(1,nsplit+1):
    raw_file = os.path.join(raw_dir, 'raw_%s_swbd_sph.%d.txt' %(feattype,i) )
    raw_lines = open(raw_file).readlines()
    sindices = [i for i,x in enumerate(raw_lines) if 'sw' in x]
    eindices = sindices[1:] + [len(raw_lines)]
    for start_idx, end_idx in zip(sindices, eindices):
        feat_dict = {}
        filename = raw_lines[start_idx].strip('[\n').rstrip().replace('sw0', 'sw')
        if filename in done_files:
            print "already done: ", filename
            continue
        frames = raw_lines[start_idx+1:end_idx]
        list_feats = [f.strip().split()[:numc] for f in frames]
        floated_feats = [[float(x) for x in coef] for coef in list_feats]
        feat_dict[filename] = floated_feats
        full_name = os.path.join(output_dir, filename + '.pickle') 
        pickle.dump(feat_dict, open(full_name, 'w'))


