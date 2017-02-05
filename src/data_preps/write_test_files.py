import os
import sys
import glob
import cPickle as pickle
from tree_utils import *


# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]


def process_data(data_dir, split):
    data_set = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            sent_ids = sentence.lower().split()
            parse_ids = parse.split() 
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)+1]
            # +1 to account for EOS in bucketing
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            tree = merge_sent_tree(parse_ids, sent_ids)
            data_set[bucket_id].append([sent_ids, parse_ids, tree])
    return data_set

def map_sentences(data_dir, split):
    mappings = [[] for _ in _buckets]
    split_path = os.path.join(data_dir, split)
    split_files = glob.glob(split_path + "/*")
    for file_path in split_files:
        this_data = pickle.load(open(file_path))
        for k in this_data.keys():
            sentence = this_data[k]['sents']
            parse = this_data[k]['parse']
            sent_ids = sentence.rstrip().split() 
            parse_ids = parse.rstrip().split()
            #include line below for swbd_new, but not for swbd_speech and swbd_tune
            if split != 'extra':
                parse_ids.append(EOS_ID)
            maybe_buckets = [b for b in xrange(len(_buckets)) 
                if _buckets[b][0] >= len(sent_ids) and _buckets[b][1] >= len(parse_ids)]
                #if _buckets[b][0] > len(sent_ids) and _buckets[b][1] > len(parse_ids)]
            # > for swbd_speech; >= for swbd_new
            if not maybe_buckets: 
                #print(k, sentence, parse)
                continue
            bucket_id = min(maybe_buckets)
            mappings[bucket_id].append(k)
    return mappings 

    

if __name__ == "__main__":
    data_dir = "/scratch/ttran/Datasets/swbd_speech"
    out_dir = "/scratch/ttran/"
    split = 'test'

    sentfile = os.path.join(out_dir, split+'.sents')
    treefile = os.path.join(out_dir, split+'.trees')
    fsent = open(sentfile, 'w')
    ftree = open(treefile, 'w')

    data_set = process_data(data_dir, split)
    for bucket in data_set:
        for sample in bucket:
            sent_ids, parse_ids, tree = sample
            fsent.write('{}\n'.format(' '.join(sent_ids)))
            ftree.write('{}\n'.format(' '.join(tree)))
    fsent.close()
    ftree.close()


