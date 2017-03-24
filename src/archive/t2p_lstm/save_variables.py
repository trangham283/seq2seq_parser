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
import argparse
import data_utils
import seq2seq_model
from tree_utils import add_brackets, match_length, delete_empty_constituents, merge_sent_tree

pa = argparse.ArgumentParser(description='Evaluate swbd model')
pa.add_argument('num_load', help='model step number to load')
args = pa.parse_args()
num_load = args.num_load

train_dir = '/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-lstm-0905'
model_path = os.path.join(train_dir, 'parse_nn_small.ckpt-' + num_load)

# default params
learning_rate = 0.1
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
hidden_size = 256
embedding_size = 512
num_layers = 3
input_vocab_size = 90000
output_vocab_size = 128
batch_size = 128
dropout=False
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1


def create_model_default(session, forward_only, dropout=dropout, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      input_vocab_size, output_vocab_size, _buckets,
      hidden_size, num_layers, embedding_size,
      max_gradient_norm, batch_size,
      learning_rate, learning_rate_decay_factor,
      forward_only=forward_only, dropout=dropout)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not model_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
    print("loaded from %d done steps" %(steps_done) )
  elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and model_path is not None:
    model.saver.restore(session, model_path)
    steps_done = int(model_path.split('-')[-1])
    print("Reading model parameters from %s" % model_path)
    print("loaded from %d done steps" %(steps_done) )
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
    steps_done = 0
  return model, steps_done


def save_vars(filename):
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
      model_dev, steps_done = create_model_default(sess, forward_only=True, dropout=False, model_path=model_path)
    

    var_dict = {}
    for var in tf.all_variables():
      print(var.name, var.get_shape())
      if 'Adagrad' in var.name: continue
      var_dict[var.name] = var.eval()

    pickle.dump(var_dict, open(filename, 'w'))

    #for v in tf.all_variables():
    #  print(v.name, v.get_shape())


if __name__ == "__main__":
    filename = 'variables-step' + num_load + '.pickle'
    save_vars(filename) 


