#!/bin/env python

"""Convert the Switchboard corpus via the NXT XML annotations, instead of the Treebank3
format. The difference is that there's no issue of aligning the dps files etc."""

# Trang's edits: 
#   * keep everything except traces
#   * might have version where fillers/dfl are all removed
#   * a few other edits to match formatting of linearizing tree scripts

import os
import sys
import re
import nltk
import Treebank.PTB
import pandas as pd
import numpy as np
import cPickle as pickle
from itertools import izip
from tree_utils import linearize_tree
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_UNF = b"_UNF"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _UNF]

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def get_dfl(word, sent):
    turn = '%s%s' % (sent.speaker, sent.turnID[1:])
    dfl = [turn, '1' if word.isEdited() else '0', str(word.start_time), str(word.end_time)]
    return '|'.join(dfl)


def speechify(sent):
    for word in sent.listWords():
        if word.parent() is None or word.parent().parent() is None:
            continue
        if word.text == '?':
            word.prune()
        if word.isPunct() or word.isTrace() or word.isPartial():
            word.prune()
        word.text = word.text.lower()


def remove_repairs(sent):
    for node in sent.depthList():
        if node.label == 'EDITED':
            node.prune()


def remove_fillers(sent):
    for word in sent.listWords():
        if word.label == 'UH':
            word.prune()



def remove_prn(sent):
    for node in sent.depthList():
        if node.label != 'PRN':
            continue
        words = [w.text for w in node.listWords()]
        if words == ['you', 'know'] or words == ['i', 'mean']:
            node.prune()


def prune_empty(sent):
    for node in sent.depthList():
        if not node.listWords():
            try:
                node.prune()
            except:
                print node
                print sent
                raise


def prune_trace(sent):
    for word in sent.listWords():
        if word.isTrace(): word.prune()

def cleanup(sent):
    for word in sent.listWords():
        if word.text == '?' or word.text == '!':
            word.prune()
        if word.isPunct() or word.isTrace(): 
            word.prune()
        word.text = word.text.lower()


# detach immediate bracket from terminal for later linearize_tree formatting
def detach_brackets(line):
    items = line.strip().split()
    sent = []
    for tok in items:
        num_close = tok.count(')')
        if num_close <= 1 or tok[0] == ')':
            sent.append(tok)
        else:
            word = tok.strip(')')
            sent.append(word+')')
            leftover = ''.join([')']*(num_close - 1))
            sent.append(leftover)
    return sent

def write_txt(txt_dir, out_dir):
    train_file = os.path.join(txt_dir, 'train.data_with_punctuations.csv')
    text_file = os.path.join(out_dir, 'train_sents_punctuations.txt')
    parse_file = os.path.join(out_dir, 'train_parse_punctuations.txt')
    df = pd.read_csv(train_file, sep='\t') 
    ft = open(text_file, 'w')
    fp = open(parse_file, 'w')

    for i, row in df.iterrows():
        ft.write(row.sentence.lower() + '\n')
        fp.write(row.parse + '\n') 
    
    ft.close()
    fp.close()

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, \
        tokenizer=None, normalize_digits=True):
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

def do_section_keep_punc(ptb_files, out_dir, name):
    times_file = os.path.join(out_dir, '%s.data_with_punctuations.csv' % name)
    list_row = []
    for file_ in ptb_files:
        sents = []
        orig_words = []
        for sent in file_.children():
            prune_trace(sent)
            line = str(sent)
            new_tree = detach_brackets(line)
            # Weird non-sentence case:
            if len(new_tree) <= 1: continue
            sentence, parse = linearize_tree(new_tree)
            assert len(sent._children) == 1
            stimes = [w.start_time for w in sent.listWords()] 
            etimes = [w.end_time for w in sent.listWords()] 
            words = [w.text for w in sent.listWords()]
            assert len(stimes) == len(etimes) == len(words)
            if not stimes or not etimes:
                print "no time info for ", sent.globalID
                continue
            list_row.append({'file_id': file_.ID, \
                    'speaker': sent.speaker, \
                    'sent_id': sent.globalID, \
                    'tokens': words, \
                    'start_times': stimes, \
                    'end_times': etimes, \
                    'sentence': sentence, \
                    'parse': parse})
    data_df = pd.DataFrame(list_row)
    data_df.to_csv(times_file, sep='\t', index=False)

def do_section(ptb_files, out_dir, name):
    times_file = os.path.join(out_dir, '%s.data.csv' % name)
    list_row = []
    for file_ in ptb_files:
        sents = []
        orig_words = []
        for sent in file_.children():
            cleanup(sent)
            line = str(sent)
            new_tree = detach_brackets(line)
            # Weird non-sentence case:
            if len(new_tree) <= 1: continue
            sentence, parse = linearize_tree(new_tree)
            assert len(sent._children) == 1
            stimes = [w.start_time for w in sent.listWords()] 
            etimes = [w.end_time for w in sent.listWords()] 
            words = [w.text for w in sent.listWords()]
            assert len(stimes) == len(etimes) == len(words)
            if not stimes or not etimes:
                print "no time info for ", sent.globalID
                continue
            list_row.append({'file_id': file_.ID, \
                    'speaker': sent.speaker, \
                    'sent_id': sent.globalID, \
                    'tokens': words, \
                    'start_times': stimes, \
                    'end_times': etimes, \
                    'sentence': sentence, \
                    'parse': parse})
    data_df = pd.DataFrame(list_row)
    data_df.to_csv(times_file, sep='\t', index=False)


def get_stats(ptb_files, out_dir, name):
    dict_file = os.path.join(out_dir, '%s.pos-tag.pickle' % name)
    pos_dict = dict()
    for file_ in ptb_files:
        sents = []
        orig_words = []
        for sent in file_.children():
            assert len(sent._children) == 1
            for w in sent.listWords():
                pos = w.label
                if pos not in pos_dict:
                    pos_dict[pos] = 0
                pos_dict[pos] += 1

    pickle.dump(pos_dict, open(dict_file, 'w'))
    for k in sorted(pos_dict.keys()):
        print k, pos_dict[k] 

def get_prns(ptb_files, split):
    counts = {}
    counts['PRN'] = 0
    counts['nonPRN'] = 0
    for file_ in ptb_files:
        for sent in file_.children():
            nodes_by_depth = []
            for node in sent.depthList():
                words = [w.text for w in node.listWords()]
                if words == ['you', 'know'] or words == ['i', 'mean']:
                    if node.label != "PRN" and node.parent().label != "PRN":
                        print sent.globalID, node.label, words
                        print sent
                        counts['nonPRN'] += 1
                    if node.label == 'PRN':
                        counts['PRN'] += 1
                    if node.label == 'S' and node.parent().label == 'PRN':
                        counts['PRN'] += 1
    print split, counts

def prep_bkparser_data(ptb_files, out_dir, name):
    out_file = os.path.join(out_dir, '%s_trees_for_bk.mrg' % name)
    out_f = open(out_file, 'w')
    for file_ in ptb_files:
        for sent in file_.children():
            assert len(sent._children) == 1
            stimes = [w.start_time for w in sent.listWords()] 
            etimes = [w.end_time for w in sent.listWords()] 
            words = [w.text for w in sent.listWords()]
            assert len(stimes) == len(etimes) == len(words)
            if not stimes or not etimes:
                print "no time info for ", sent.globalID
                continue
            prune_trace(sent)
            prune_empty(sent)
            line = str(sent)
            new_tree = detach_brackets(line)
            if len(new_tree) <= 1: continue
            new_tree = ['(ROOT'] + new_tree + [')']
            item = ' '.join(new_tree) + '\n'
            out_f.write(item)
    out_f.close()

def prep_bkparser_eval(ptb_files, out_dir, name):
    out_file = os.path.join(out_dir, '%s_sents_for_bk.txt' % name)
    out_f = open(out_file, 'w')
    for file_ in ptb_files:
        for sent in file_.children():
            assert len(sent._children) == 1
            stimes = [w.start_time for w in sent.listWords()] 
            etimes = [w.end_time for w in sent.listWords()] 
            words = [w.text for w in sent.listWords()]
            assert len(stimes) == len(etimes) == len(words)
            if not stimes or not etimes:
                print "no time info for ", sent.globalID
                continue
            prune_trace(sent)
            prune_empty(sent)
            line = str(sent)
            new_tree = detach_brackets(line)
            if len(new_tree) <= 1: continue
            sentence, parse = linearize_tree(new_tree)
            item = sentence + '\n'
            out_f.write(item)
    out_f.close()

def main(nxt_loc, out_dir):
    corpus = Treebank.PTB.NXTSwitchboard(path=nxt_loc)
    #prep_bkparser_data(corpus.dev2_files(), out_dir, 'dev2')
    #prep_bkparser_data(corpus.dev_files(), out_dir, 'dev')
    #prep_bkparser_data(corpus.eval_files(), out_dir, 'test')
    #prep_bkparser_data(corpus.train_files(), out_dir, 'train')

    #prep_bkparser_eval(corpus.dev2_files(), out_dir, 'dev2')
    prep_bkparser_eval(corpus.dev_files(), out_dir, 'dev')
    prep_bkparser_eval(corpus.eval_files(), out_dir, 'test')
    #prep_bkparser_eval(corpus.train_files(), out_dir, 'train')

    #get_prns(corpus.train_files(), 'train')
    #get_prns(corpus.dev_files(), 'dev')
    #get_prns(corpus.dev2_files(), 'dev2')
    #get_prns(corpus.eval_files(), 'test')
    
    #do_section(corpus.train_files(), out_dir, 'train')
    #do_section(corpus.dev_files(), out_dir, 'dev')
    #do_section(corpus.dev2_files(), out_dir, 'dev2')
    #do_section(corpus.eval_files(), out_dir, 'test')
    
    #get_stats(corpus.train_files(), out_dir, 'train')
    #get_stats(corpus.dev_files(), out_dir, 'dev')
    #get_stats(corpus.dev2_files(), out_dir, 'dev2')
    #get_stats(corpus.eval_files(), out_dir, 'test')

    #do_section_keep_punc(corpus.train_files(), out_dir, 'train')
    #do_section_keep_punc(corpus.dev_files(), out_dir, 'dev')
    #do_section_keep_punc(corpus.dev2_files(), out_dir, 'dev2')
    #do_section_keep_punc(corpus.eval_files(), out_dir, 'test')

    #write_txt(out_dir, out_dir)
    #sent_data_path = os.path.join(out_dir, 'train_sents_punctuations.txt')
    #parse_data_path = os.path.join(out_dir, 'train_parse_punctuations.txt')
    #sent_vocabulary_path = os.path.join(out_dir, 'vocab_punc.sents')
    #parse_vocabulary_path = os.path.join(out_dir, 'vocab_punc.parse')
    #create_vocabulary(sent_vocabulary_path, sent_data_path, 45000)
    #create_vocabulary(parse_vocabulary_path, parse_data_path, 45000)

if __name__ == '__main__':
    nxt_loc = '/s0/ttmt001/speech_parsing'
    out_dir = '/s0/ttmt001/speech_parsing/swbd_trees'
    main(nxt_loc, out_dir)
