#!/usr/bin/env python
# Tree utils for completing brackets and filling out missing words 


import os
import sys
import argparse
import random
import re

def add_brackets(toks):
    line = ' '.join(toks)
    num_open = line.count('(')
    num_close = line.count(')')
    if num_open == num_close:
        full_sent = toks[:]
        valid = 1
    else:
        valid = 0
        if num_open < num_close:
            add_open = num_close - num_open
            extra_open = ['(']*add_open
            full_sent = extra_open + toks
        else:
            add_close = num_open - num_close
            extra_close = [')']*add_close
            full_sent = toks + extra_close
    return full_sent, valid



def match_length(parse, sent):
    line = ' '.join(parse)
    PUNC = ['.', ',', ':', '``', '\'\'', ';', '?', '!', '$', '"', '%', '*', '&']
    tree = []
    sent_toks = sent[:]
    dec_toks = parse[:]
    num_toks = len(sent_toks)
    num_parse = line.count('XX') 
    num_puncs = sum([line.count(x) for x in PUNC])
    num_out = num_puncs + num_parse
    if num_toks == num_out:
        new_tree = dec_toks[:]
    else:
        if num_out < num_toks: # add 'XX' in this case
            num_X = num_toks - num_out  
            for _ in range(num_X):
                if len(dec_toks) > 3:
                    x_add = random.choice(range(len(dec_toks) - 2)) 
                    # offset a bit so never insert at very beginning or very end
                    dec_toks.insert(x_add + 2, 'XX')
                else:
                    dec_toks.insert(1, 'XX')
            new_tree = dec_toks[:]
        else: # remove XXs 
            num_X = num_out - num_toks
            x_indices = [i for i, x in enumerate(dec_toks) if x == "XX"]
            if num_X < len(x_indices):
                x_remove = random.sample(set(x_indices), num_X)
                for k in x_remove:
                    dec_toks[k] = "TO_DELETE"
                for _ in range(len(x_remove)):
                    dec_toks.remove("TO_DELETE")
            # else: do nothing
            new_tree = dec_toks[:]
    return new_tree



def merge_sent_tree(parse, sent):
    tree = []
    word_idx = 0
    for token in parse:
        tok = token
        if token == 'XX': 
            if word_idx < len(sent):
                tok = '(XX {})'.format(sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        elif token[0] == ')':
            tok = ')'
        elif token[0] != '(':
            if word_idx < len(sent):
                tok = '({} {})'.format(token, sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        tree.append(tok)
    new_tree = []
    idx = 0
    k = 0
    while idx < len(tree):
        token = tree[idx]
        if token == ')':
            k = 1
            while (idx + k) < len(tree):
                if tree[idx+k] != ')':
                    break
                k += 1
            token = ')' * k
            idx += k - 1
        idx += 1
        new_tree.append(token)

    return new_tree


