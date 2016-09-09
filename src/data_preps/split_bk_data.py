import os
import random
import sys

import numpy as np
import tensorflow as tf
import cPickle as pickle

tf.app.flags.DEFINE_string("data_dir", "/home-nfs/ttran/transitory/for_batch_jobs/swbd_data/", "directory of files")
tf.app.flags.DEFINE_integer("num_splits", 20, "number of splits to make")
FLAGS = tf.app.flags.FLAGS


def split_data(data_path, bucket_id):
    basename = os.path.basename(data_path).rstrip('.pickle')
    data = pickle.load(open(data_path))
    print len(data)
    np.random.shuffle(data)
    start_idx = np.arange(0, len(data), len(data) / (FLAGS.num_splits) )
    if len(start_idx) > FLAGS.num_splits:
        end_idx = np.ones(FLAGS.num_splits + 1, dtype=int) * len(data) 
        end_idx[:-1] = start_idx[1:]
        # merge last split since it has very few sentences
        start_idx = start_idx[:-1]
        end_idx[-2] = end_idx[-1]
        end_idx = end_idx[:-1]
    else:
        end_idx = np.ones(FLAGS.num_splits, dtype=int) * len(data)
        end_idx[:-1] = start_idx[1:]


    for i in range(len(start_idx)):
        new_name = os.path.join(FLAGS.data_dir, basename + '-split-' + str(i) + '.pickle')
        this_data = data[start_idx[i]:end_idx[i]]
        print "File: ", basename + '-split-' + str(i), "; size: ", len(this_data)
        pickle.dump(this_data, open(new_name, 'w'))

    
def main(_):
    data_dir = FLAGS.data_dir
    #for bucket_id in range(3):
    for bucket_id in [2]:
        bk_path = os.path.join(data_dir, 'bktotal.train.data.set-' + str(bucket_id) + '.pickle')
        split_data(bk_path, bucket_id)


if __name__ == "__main__":
    tf.app.run()
