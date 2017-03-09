"""Convert the Switchboard corpus via the NXT XML annotations, instead of the Treebank3
format. The difference is that there's no issue of aligning the dps files etc."""

# Trang's edits: 
#   * keep everything except traces
#   * might have version where fillers/dfl are all removed
#   * a few other edits to match formatting of linearizing tree scripts

import os
import sys
import Treebank.PTB
import plac

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
    trees_file = os.path.join(out_dir, '%s.trees' % name)
    times_file = os.path.join(out_dir, '%s.times' % name)
    trees = open(trees_file, 'w')
    times = open(times_file, 'w')
    err_sents = []
    p2_sents = []
    for file_ in ptb_files:
        sents = []
        orig_words = []
        for sent in file_.children():
            prune_trace(sent)
            assert len(sent._children) == 1
            stimes = [w.start_time for w in sent.listWords() if w.start_time is not None]
            etimes = [w.end_time for w in sent.listWords() if w.end_time is not None]
            if not stimes or not etimes:
                err_sents.append(sent.globalID)
                continue
                
            start_time = stimes[0]
            end_time = etimes[-1]
            if start_time == -1 or end_time == -1:
                # skip non-aligned sentence
                continue
            
            # check if utterance contains multiple sentences
            w = [c.text for c in sent.listWords()]
            indices = [i for i, x in enumerate(w) if x == "."]
            if len(indices) > 1:
                p2_sents.append(sent.globalID)
                
                #if indices[-1] == len(w) - 1: # last period is last in sentence
                #    start_indices = [0] + [k-1 for k in indices[:-1]] 
                #    end_indices = [k+1 for k in indices]
                #else:
                #    start_indices = [0] + [k-1 for k in indices]
                #    end_indices = [k+1 for k in indices] + [len(w)]
                #for s, e in zip(start_indices, end_indices):
                #    sub_sent = sent[s:e]
            line = str(sent)
            new_tree = detach_brackets(line)
            trees.write('{}\n'.format(' '.join(new_tree)))
            # write timing info
            # filename \t speaker \t globalID \t start_time \t end_time \t length \n
            item = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(file_.filename, sent.speaker, \
                sent.globalID, start_time, end_time, len(w) )
            times.write(item)

    print "Sentences without time-alignment: "
    for s in err_sents: print s

    print "Sentence with more than 1 period: "
    for s in p2_sents: print s


def main(nxt_loc, out_dir):
    corpus = Treebank.PTB.NXTSwitchboard(path=nxt_loc)
    #do_section(corpus.train_files(), out_dir, 'train')
    do_section(corpus.dev_files(), out_dir, 'dev')
    #do_section(corpus.dev2_files(), out_dir, 'dev2')
    #do_section(corpus.eval_files(), out_dir, 'test')


if __name__ == '__main__':
    plac.call(main)
