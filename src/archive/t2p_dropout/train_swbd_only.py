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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""

# ttmt update: 
#   changed name to parse_nn.py
#   use data_utils and seq2seq model specific in this directory

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

#from tensorflow.models.rnn.translate import data_utils
#from tensorflow.models.rnn.translate import seq2seq_model
import data_utils
import seq2seq_model
import subprocess
from tree_utils import add_brackets, match_length, merge_sent_tree


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("dropout", True, "To use dropout or not.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 512, "Size embeddings.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("input_vocab_size", 90000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("output_vocab_size", 128, "output vocabulary size.")
tf.app.flags.DEFINE_integer("max_steps", 500, "max number of steps, in terms of batch passes.")
tf.app.flags.DEFINE_string("data_dir", "/share/data/speech/Data/ttran/for_batch_jobs/swbd_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "checkpoint frequency, in terms of steps")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding a given file.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS

# Max number of steps safely done in 4hr limit
# max_sess_steps = FLAGS.max_steps

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

# data set paths
dev_data_path = os.path.join(FLAGS.data_dir, 'swbd.dev.set.pickle')
swtrain_data_path = os.path.join(FLAGS.data_dir, 'swbd.train.set.pickle')
dev_set = pickle.load(open(dev_data_path))
train_sw = pickle.load(open(swtrain_data_path))

train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
print(train_bucket_sizes)
train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
offset_lengths = [len(x) for x in train_bucket_offsets]
tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
all_bucks = [x for sublist in tiled_buckets for x in sublist]
all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
train_set = zip(all_bucks, all_offsets)

np.random.shuffle(train_set)


# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

def create_model(session, forward_only, dropout, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, dropout=dropout)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
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

def train():
  """Train a sequence to sequence parser."""
  # Prepare data
  print("Loading data from %s" % FLAGS.data_dir)

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, forward_only=False, dropout=True)
    with tf.variable_scope("model", reuse=True):
      model_dev = seq2seq_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=True, dropout=False)

    num_remaining_steps = FLAGS.max_steps - steps_done
    # This is the training loop. 
    sys.stdout.flush()
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    epoch = 0

    while current_step <= num_remaining_steps:
      epoch += 1
      print("Doing epoch: ", epoch)
      np.random.shuffle(train_set)
      for bucket_id, bucket_offset in train_set:
        #print(bucket_id, bucket_offset)
        this_sample = train_sw[bucket_id][bucket_offset:bucket_offset+FLAGS.batch_size]
        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights, seq_len = model.get_decode_batch(
                {bucket_id: this_sample}, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, seq_len, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          # Print statistics for the previous epoch.
          perplexity = math.exp(loss) if loss < 300 else float('inf')
          print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(FLAGS.train_dir, "parse_nn_small.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
          step_time, loss = 0.0, 0.0
          
          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights, seq_len = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, seq_len, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          print("Do EVALB outside separately")
          sys.stdout.flush()
    

def self_test():
  """Test the translation model."""
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    print("Self-test for modified neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 2, 2, 10,
                                       5.0, 2, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      for bucket_id in (0, 1):
        for bucket_offset in (0,2):
          encoder_inputs, decoder_inputs, target_weights, seq_len = model.get_ordered_batch(
                  data_set, bucket_id, bucket_offset)
          model.step(sess, encoder_inputs, decoder_inputs, target_weights, seq_len, bucket_id, False)
          print(len(encoder_inputs), len(decoder_inputs))



def main(_):
  if FLAGS.self_test:
    self_test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
