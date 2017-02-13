"""Convert the Switchboard corpus via the NXT XML annotations, instead of the Treebank3
format. The difference is that there's no issue of aligning the dps files etc."""

# Trang's edits: 
#   * keep everything except traces
#   * might have version where fillers/dfl are all removed
#   * a few other edits to match formatting of linearizing tree scripts

import os
import sys
import Treebank.PTB
import pandas as pd
from itertools import izip

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
        if word.text == '?':
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


def do_section(ptb_files, out_dir, name):
    times_file = os.path.join(out_dir, '%s.data.csv' % name)
    list_row = []
    for file_ in ptb_files:
        sents = []
        orig_words = []
        for sent in file_.children():
            prune_trace(sent)
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
                    'end_times': etimes})
    data_df = pd.DataFrame(list_row)
    data_df.to_csv(times_file, sep='\t', index=False)
             

def main(nxt_loc, out_dir):
    corpus = Treebank.PTB.NXTSwitchboard(path=nxt_loc)
    do_section(corpus.train_files(), out_dir, 'train')
    do_section(corpus.dev_files(), out_dir, 'dev')
    #do_section(corpus.dev2_files(), out_dir, 'dev2')
    do_section(corpus.eval_files(), out_dir, 'test')


if __name__ == '__main__':
    nxt_loc = '/s0/ttmt001/speech_parsing'
    out_dir = '/s0/ttmt001/speech_parsing/swbd_parses'
    main(nxt_loc, out_dir)
