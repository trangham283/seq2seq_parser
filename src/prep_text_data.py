# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# ttmt update: 
#   changed name to parse_nn.py
#   use data_utils and seq2seq model specific in this directory
#   this file only uses the read_data portion of the code to
#   preprocess data into buckets

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cPickle as pickle

tf.app.flags.DEFINE_string("data_dir", "/tmp/", "directory of files")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
FLAGS = tf.app.flags.FLAGS

# Use the following buckets: 
EOS_ID = 2
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language (sentences, maybe reversed).
    target_path: path to the file with token-ids for the target language (parse tree linearized as sequences);
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    data_set_dict: dictionary to store original lines of data in the bucket-ed 
      set; i.e. data_set_dict[bucket_id] stores line indices in original raw 
      data file, corresponding to entries in the data_set array
  """
  data_set = [[] for _ in _buckets]
  data_set_dict = [[] for _ in _buckets]
  line_num = 0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            data_set_dict[bucket_id].append(line_num)
            break
        source, target = source_file.readline(), target_file.readline()
        line_num += 1
  return data_set, data_set_dict


def prep_data(sents_ids_path, parse_ids_path):
    data_set, data_set_dict = read_data(sents_ids_path, parse_ids_path)
    basename = os.path.basename(parse_ids_path)[:-13]
    data_set_name = os.path.join(FLAGS.data_dir, basename + '.set.pickle')
    data_dict_name = os.path.join(FLAGS.data_dir, basename + '.dict.pickle')
    pickle.dump(data_set, open(data_set_name, 'w'), protocol=-1)
    pickle.dump(data_set_dict, open(data_dict_name, 'w'), protocol=-1)
    
    
def main(_):
    data_dir = FLAGS.data_dir
    sw_train_ids_sents = os.path.join(data_dir, 'swbd.train.data.ids90000.sents') 
    sw_train_ids_parse = os.path.join(data_dir, 'swbd.train.data.ids128.parse')
    sw_dev_ids_sents = os.path.join(data_dir, 'swbd.dev.data.ids90000.sents') 
    sw_dev_ids_parse = os.path.join(data_dir, 'swbd.dev.data.ids128.parse')
    sw_dev2_ids_sents = os.path.join(data_dir, 'swbd.dev2.data.ids90000.sents') 
    sw_dev2_ids_parse = os.path.join(data_dir, 'swbd.dev2.data.ids128.parse')
    #sw_test_ids_sents = os.path.join(data_dir, 'swbd.test.data.ids90000.sents') 
    #sw_test_ids_parse = os.path.join(data_dir, 'swbd.test.data.ids128.parse')
    bk_ids_sents = os.path.join(data_dir, 'bktotal.train.data.ids90000.sents') 
    bk_ids_parse = os.path.join(data_dir, 'bktotal.train.data.ids128.parse')
    #prep_data(sw_dev_ids_sents, sw_dev_ids_parse)
    #prep_data(sw_dev2_ids_sents, sw_dev2_ids_parse)
    #prep_data(sw_test_ids_sents, sw_test_ids_parse)
    #prep_data(sw_train_ids_sents, sw_train_ids_parse)
    prep_data(bk_ids_sents, bk_ids_parse)


if __name__ == "__main__":
    tf.app.run()
