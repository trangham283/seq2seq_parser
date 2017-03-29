#!/bin/env python

import os
import glob
import pandas
import cPickle as pickle
from tree_utils import *
from scipy import stats
from nlp_util import pstree_plus
import numpy as np
import matplotlib.pyplot as plt

base_dir = '/s0/ttmt001/speech_parsing/'
pause_dir = os.path.join(base_dir, 'pause_data')
data_dir = os.path.join(base_dir, 'word_level')

def make_dict(pause_data):
    pauses = dict()
    for bucket in pause_data:
        for sample in bucket:
            sent_id, info_dict, parse = sample
            pauses[sent_id] = info_dict
    return pauses

def process_split(split):
    pause_file = os.path.join(pause_dir, split+'_nopunc.pickle')
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")

    pause_data = pickle.load(open(pause_file))
    pauses = make_dict(pause_data)

    list_row = []
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
            sent_toks = sentence.strip().split()
            parse_toks = parse.strip().split()
            merged = merge_sent_tree(parse_toks, sent_toks)
            merged = ' '.join(merged)
            tree = pstree_plus.tree_from_text(merged)
            tree.add_pauses(pause_bef, pause_aft)
            for node in tree:
                if node.label == 'XX':
                    continue
                span = node.span
                span_length = span[1] - span[0]
                start_word = tree.get_nodes('lowest', span[0], span[0]+1)
                end_word = tree.get_nodes('lowest', span[1]-1, span[1])
                pause_00 = start_word.pause_before
                pause_01 = start_word.pause_after
                pause_10 = end_word.pause_before
                pause_11 = end_word.pause_after
                list_row.append({'label': node.label, \
                        'span': span, \
                        'span_length': span_length, \
                        'p00': pause_00, \
                        'p01': pause_01, \
                        'p10': pause_10, \
                        'p11': pause_11})

    summary = os.path.join(base_dir, split+'_constituent_data.csv')
    list_df = pandas.DataFrame(list_row)
    list_df.to_csv(summary, sep='\t', index=False)

# 0 = _PAD
# 1 = off
# 2 = na
# 3 = 1, ie <=0.05
# 4 = 2, ie <=0.2
# 5 = 3, ie <=1 
# 6 = 4, ie the remaining
def plot_hist(y, bins, title):
    centers = [0.5, 1.5, 2.5, 3.5, 4.5, 7.5, 12.5, 17.5, 25.0, 60]
    widths = [0.9, 0.9, 0.9, 0.9, 0.9, 4.9, 4.9, 4.9, 9.9, 59.9]
    counts, edges = np.histogram(y, bins=bins)
    plt.bar(centers, counts, widths)
    plt.ylabel('constituent counts', fontsize=12)
    plt.xlabel('span length', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()


def plot_pause(y, title):
    bins = [0,1,2,3,4,5,6,7]
    counts, edges = np.histogram(y, bins=bins)
    
def comp_corr(df, ptype):
    if ptype=='begin':
        valid_df = df[(df.p00 >2)]
    else:
        valid_df = df[(df.p11 >2)]
    lengths = valid_df.span_length.values
    if ptype == 'begin':
        plengths = valid_df.p00.values
    else:
        plengths = valid_df.p11.values
    print float(len(valid_df))/len(df), '\t', stats.spearmanr(plengths, lengths)[0]


comp_corr(df, 'begin')
comp_corr(df_sub, 'begin')

comp_corr(df_NP, 'begin')
comp_corr(df_VP, 'begin')

def analyze_split(split):
    summary = os.path.join(base_dir, split+'_constituent_data.csv')
    df = pandas.read_csv(summary, sep='\t')
    df_sub = df[df.label != 'S']
    bins = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 90]
    max_value = df_sub.span_length.max()
    rights = bins[1:] + [max_value]
    #centers = [(x+y)*0.5 for x, y in zip(bins, rights)]
    #widths = [0.9*(y-x) for x, y in zip(bins, rights)]
    df_NP = df_sub[df_sub.label=='NP']
    y = df_NP.span_length.values
    title = 'NP constituents'
    df_VP = df_sub[df_sub.label=='VP']
    y = df_VP.span_length.values
    title = 'VP constituents'
    df_PP = df_sub[df_sub.label=='PP']
    y = df_PP.span_length.values
    title = 'PP constituents'
    df_SBAR = df_sub[df_sub.label=='SBAR']
    y = df_SBAR.span_length.values
    title = 'SBAR constituents'
    
    df_EDITED = df_sub[df_sub.label=='EDITED']
    y = df_EDITED.span_length.values
    title = 'EDITED constituents'
    
    df_INTJ = df_sub[df_sub.label=='INTJ']
    y = df_INTJ.span_length.values
    title = 'INTJ constituents'

    df_PRN = df_sub[df_sub.label=='PRN']
    y = df_PRN.span_length.values
    title = 'PRN constituents'

# constituent counts: (excluding S)
# 628,476 total
# 222,153 NP
# 159,436 VP
#  45,571 PP
#  30,594 SBAR 
#  22,131 EDITED
#  56,010 INTJ
#   9,762 PRN
# remaining: 82822
# all: set(['EDITED', 'SBAR', 'NAC', 'SINV', 'ADJP', 'WHADVP', 'CONJP', 'PP', 'UCP', 'PRN', 'RRC', 'NX', 'PRT', 'LST', 'NP', 'WHPP', 'FRAG', 'VB', 'INTJ', 'VP', 'SBARQ', 'X', 'ADVP', 'QP', 'SQ', 'WHADJP', 'WHNP', 'TYPO'])



if __name__ == "__main__":
    #process_split('train')
    analyze_split('train')


'''
# for debugging:
# a shorter sentence:
merged = '(S (S (NP (XX we) ) (VP (XX saw) (NP (XX a) (XX husband) (XX and) (XX wife) ))))'
pause_bef = [2, 1, 2, 2, 1, 1]
pause_aft = [1, 2, 2, 1, 1, 2]



merged = "(S (S (CONJP (XX and) (XX so) ) (NP (XX it) ) (VP (XX looks) (PP (XX to) (NP (XX me) )) (SBAR (XX like) (S (ADVP (XX maybe) ) (S (NP (XX their) ) (XX not) (VP (XX catering) (PP (XX to) (NP (NP (XX this) (XX person) (XX 's) ) (XX needs) )))) (VP (XX is) (ADVP (XX really) ) (SBAR (XX because) (S (NP (XX this) (XX person) ) (EDITED (VP (XX is) (ADVP (XX just) ))) (PRN (S (NP (XX you) ) (VP (XX know) ))) (VP (XX is) (ADVP (XX just) ) (PP (XX in) (NP (NP (XX a) (XX state) ) (SBAR (WHADVP (XX where) ) (S (NP (XX they) ) (VP (XX do) (XX n't) (ADVP (XX really) ) (VP (XX need) (SBAR (WHNP (XX what) ) (S (NP (XX they) ) (VP (XX think) (SBAR (S (NP (XX they) ) (VP (XX need) )))))))))))))))))) (PRN (S (NP (XX you) ) (VP (XX know) ))))))"
pause_bef = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 4, 1]
pause_aft = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2]
'''

# 



