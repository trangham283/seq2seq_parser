#!/usr/bin/env python
# 
# File Name : linearize_tree.py
#
# Description :
#
# Usage :
#
# Creation Date : 22-10-2015
# Last Modified : Thu 12 Nov 2015 10:58:25 PM PST
# Author : Hao Cheng

import os
import sys
import argparse

def linearize_tree(args):
    keep_set = ['-NONE-', ',', ':', '``', '\'\'', '.']
    with open(args.infile) as fin,\
            open(args.outfile, 'w') as fout:

        for line in fin:
            items = line.strip().split()
            sent = []
            tree = []
            tag_stack = []
            for idx, token in enumerate(items):
                token = token.strip()
                if token[0] == '(':
                    # a tree part
                    next_token = items[idx + 1]
                    if next_token[0] == '(':
                        # not POS
                        # push the tag in stack
                        if len(token) > 1:
                            if token[1] == '(':
                                tag_stack.append('(')
                                tree.append('(')
                            tok = token.strip('(')
                            if args.rm_func_tag:
                                try:
                                    tok = tok.replace('-', ' ').replace('=', ' ').strip().split()[0]
                                except:
                                    sys.stderr.write('''Err: rm-func-tag {} token
                                            {}\n'''.format(tok, token))
                                    sys.exit(1)
                            tag_stack.append(tok)
                            tree.append('({}'.format(tok))
                        else:
                            tag_stack.append(token)
                            tree.append(token)
                    else:
                        # current is POS
                        if token[1:] == next_token[:-1] or\
                                token[1:] in keep_set:
                            tree.append(token.strip('('))
                        elif args.pos_norm:
                            tree.append('XX')
                        else:
                            tree.append(token.strip('('))
                elif token[0] == ')':
                    # bracket annotation
                    for i in range(len(token)):
                        try:
                            tag = tag_stack.pop()
                        except:
                            sys.stderr.write('Err: bracket does not match!\n')
                            sys.stderr.write('''current partial tree
                                    {}\n'''.format(' '.join(tree)))
                            sys.stderr.write('current token {}\n'.format(token))
                            sys.stderr.write('current tree {}\n'.format(line))
                            sys.exit(1)
                        if args.dec_bracket:
                            tree.append(')_{}'.format(tag))
                        else:
                            tree.append(')')
                else:
                    # word
                    if args.lower:
                        sent.append(token.strip(')').lower())
                    else:
                        sent.append(token.strip(')'))
            if args.rev_sent:
                sent.reverse()
            fout.write('{}\t{}\n'.format(' '.join(sent), ' '.join(tree)))


if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Linearize PCFG Trees')
    pa.add_argument('--infile', help='input filename')
    pa.add_argument('--outfile', help='output filename')
    pa.add_argument('--pos-norm', action='store_true', \
            dest='pos_norm', help='pos normalization to XX')
    pa.set_defaults(pos_norm=False)
    pa.add_argument('--rm-func-tag', action='store_true',\
            dest='rm_func_tag', help='remove functional tags')
    pa.set_defaults(rm_func_tag=False)
    pa.add_argument('--dec-bracket', action='store_true',\
            dest='dec_bracket', help='suffix ) with TAG')
    pa.set_defaults(dec_bracket=False)
    pa.add_argument('--lower', action='store_true',\
            dest='lower', help='lowercase the words')
    pa.set_defaults(lower=False)
    pa.add_argument('--rev-sent', action='store_true', \
            dest='rev_sent', help='reverse sent')
    pa.set_defaults(rev_sent=False)
    args = pa.parse_args()
    linearize_tree(args)
    sys.exit(0)
