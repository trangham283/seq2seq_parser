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
from tree_utils import add_brackets, match_length, merge_sent_tree, delete_empty_constituents


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("attention", False, "To use attention or not.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 512, "Size embeddings.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("input_vocab_size", 90000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("output_vocab_size", 128, "output vocabulary size.")
tf.app.flags.DEFINE_integer("max_steps", 500, "max number of steps, in terms of batch passes.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "checkpoint frequency, in terms of steps")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding a given file.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("decode_input_path", "/tmp", "path to file to be decoded")
tf.app.flags.DEFINE_string("decode_output_path", "/tmp", "path to write decoded file")
tf.app.flags.DEFINE_string("model_path", "None", "path to model for evaluation")

FLAGS = tf.app.flags.FLAGS

# Max number of steps safely done in 4hr limit
max_sess_steps = 500

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

# data set paths
dev_data_path = os.path.join(FLAGS.data_dir, 'swbd.dev.set.pickle')
swtrain_data_path = os.path.join(FLAGS.data_dir, 'swbd.train.set.pickle')
bktrain_data_paths = {}
p0 = os.path.join(FLAGS.data_dir, 'bktotal.train.data.set-0.pickle')
p1 = os.path.join(FLAGS.data_dir, 'bktotal.train.data.set-1.pickle')
p2 = os.path.join(FLAGS.data_dir, 'bktotal.train.data.set-2.pickle')
bktrain_data_paths[0] = p0
bktrain_data_paths[1] = p1
bktrain_data_paths[2] = p2

# batch size proportions
sw_batch_proportion = 0.5
sw_batch_size = int(sw_batch_proportion*FLAGS.batch_size)
bk_batch_size = FLAGS.batch_size - sw_batch_size

# choose ditribution according to sw data
#train_bucket_sizes = [61923, 24618, 3405]
#train_total_size = float(sum(train_bucket_sizes))
#train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
#                       for i in xrange(len(train_bucket_sizes))]

#offset_lengths = [len(x) for x in train_bucket_offsets]
#tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
#all_bucks = [x for sublist in tiled_buckets for x in sublist]
#all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
#ordered_train_set = zip(all_bucks, all_offsets)
#shuffled_train_set = ordered_train_set[:]
#np.random.shuffle(shuffled_train_set)

# from global step 54500 onwards, changed sampling a bit to learn on 
# longer sentences
train_buckets_scale = [0.6, 0.8, 1.0]

# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

def create_model(session, forward_only, attention=FLAGS.attention, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, attention=attention)
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
  dev_set = pickle.load(open(dev_data_path))
  train_sw = pickle.load(open(swtrain_data_path))
  
  # for now, load first bucket since it's most likely
  this_bucket_id = 0
  train_bk = pickle.load(open(bktrain_data_paths[this_bucket_id])) 

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, False)
    with tf.variable_scope("model", reuse=True):
      model_dev = seq2seq_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=True, attention=FLAGS.attention)      

    num_remaining_steps = FLAGS.max_steps - steps_done
    # This is the training loop. 
    print("Training from random batches")
    sys.stdout.flush()
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while current_step <= min(num_remaining_steps, max_sess_steps):
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs_sw, decoder_inputs_sw, target_weights_sw = model.get_mix_batch(
                train_sw[bucket_id], bucket_id, sw_batch_size)
        if bucket_id != this_bucket_id: 
            # different bucket than initial
            # load correct bucket and update this_bucket_id
            train_bk = pickle.load(open(bktrain_data_paths[bucket_id]))
            this_bucket_id = bucket_id
        encoder_inputs_bk, decoder_inputs_bk, target_weights_bk = model.get_mix_batch(
                train_bk, bucket_id, bk_batch_size)
        encoder_inputs = [np.hstack([encoder_inputs_sw[i], encoder_inputs_bk[i]]) 
                for i in range(len(encoder_inputs_sw))]
        decoder_inputs = [np.hstack([decoder_inputs_sw[i], decoder_inputs_bk[i]]) 
                                for i in range(len(decoder_inputs_sw))]
        target_weights = [np.hstack([target_weights_sw[i], target_weights_bk[i]]) 
                                for i in range(len(target_weights_sw))]

        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
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
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          print("Do EVALB outside separately")
          #do_evalb(model_dev, model, sess, dev_set)
          sys.stdout.flush()
    
def process_eval(out_lines, this_size):
  # main stuff between outlines[3:-32]
  results = out_lines[3:-32]
  assert len(results) == this_size
  matched = 0 
  gold = 0
  test = 0
  for line in results:
      m, g, t = line.split()[5:8]
      matched += int(m)
      gold += int(g)
      test += int(t)
  return matched, gold, test

      
def do_evalb(model_dev, model, sess, dev_set):  
  # Load vocabularies.
  eval_batch_size = 50
  sents_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.sents" % FLAGS.input_vocab_size)
  parse_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.parse" % FLAGS.output_vocab_size)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

  gold_file_name = os.path.join(FLAGS.train_dir, 'dev.gold.txt')
  # file with matched brackets
  decoded_br_file_name = os.path.join(FLAGS.train_dir, 'dev.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(FLAGS.train_dir, 'dev.decoded.mx.txt')
  
  print("Doing evalb")
  print("Debug - step: ", model.global_step.eval())
  
  num_sents = []
  num_valid_br = []
  num_valid_mx = []
  br = []
  mx = []

  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, eval_batch_size) 
    for batch_offset in offsets:
        fout_gold = open(gold_file_name, 'w')
        fout_br = open(decoded_br_file_name, 'w')
        fout_mx = open(decoded_mx_file_name, 'w')
        all_examples = dev_set[bucket_id][batch_offset:batch_offset + eval_batch_size]
        model_dev.batch_size = len(all_examples)        
        token_ids = [x[0] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        dec_ids = [[]] * len(token_ids)
        encoder_inputs, decoder_inputs, target_weights = model_dev.get_decode_batch(
                {bucket_id: zip(token_ids, dec_ids)}, bucket_id)
        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
        outputs = [np.argmax(logit, axis=1) for logit in output_logits]
        to_decode = np.array(outputs).T
        num_valid = 0
        num_sents.append(to_decode.shape[0])
        for sent_id in range(to_decode.shape[0]):
          parse = list(to_decode[sent_id, :])
          if data_utils.EOS_ID in parse:
            parse = parse[:parse.index(data_utils.EOS_ID)]
          # raw decoded parse
          # print(parse)
          decoded_parse = []
          for output in parse:
              if output < len(rev_parse_vocab):
                decoded_parse.append(tf.compat.as_str(rev_parse_vocab[output]))
              else:
                decoded_parse.append("_UNK") 
          # decoded_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in parse]
          # add brackets for tree balance
          parse_br, valid = add_brackets(decoded_parse)
          num_valid += valid
          # get gold parse, gold sentence
          gold_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in gold_ids[sent_id]]
          sent_text = [tf.compat.as_str(rev_sent_vocab[output]) for output in token_ids[sent_id]]
          # parse with also matching "XX" length
          parse_mx = match_length(parse_br, sent_text)
          parse_mx = delete_empty_constituents(parse_mx)

          to_write_gold = merge_sent_tree(gold_parse[:-1], sent_text) # account for EOS
          to_write_br = merge_sent_tree(parse_br, sent_text)
          to_write_mx = merge_sent_tree(parse_mx, sent_text)

          fout_gold.write('{}\n'.format(' '.join(to_write_gold)))
          fout_br.write('{}\n'.format(' '.join(to_write_br)))
          fout_mx.write('{}\n'.format(' '.join(to_write_mx)))
          
        # call evalb
        fout_gold.close()
        fout_br.close()
        fout_mx.close()  
   
        # evaluate current batch
        cmd = [evalb_path, '-p', prm_file, gold_file_name, decoded_br_file_name]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out_lines = out.split("\n")
        vv = [x for x in out_lines if "Number of Valid sentence " in x]
        s1 = float(vv[0].split()[-1])
        num_valid_br.append(s1)
        m_br, g_br, t_br = process_eval(out_lines, to_decode.shape[0])
        br.append([m_br, g_br, t_br])

        cmd = [evalb_path, '-p', prm_file, gold_file_name, decoded_mx_file_name]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out_lines = out.split("\n")
        vv = [x for x in out_lines if "Number of Valid sentence " in x]
        s2 = float(vv[0].split()[-1])
        num_valid_mx.append(s2)
        m_mx, g_mx, t_mx = process_eval(out_lines, to_decode.shape[0])
        mx.append([m_mx, g_mx, t_mx])

  br_all = np.array(br)
  mx_all = np.array(mx)
  sum_br_pre = sum(br_all[:,0]) / sum(br_all[:,2])
  sum_br_rec = sum(br_all[:,0]) / sum(br_all[:,1])
  sum_br_f1 = 2 * sum_br_pre*sum_br_rec / (sum_br_rec + sum_br_pre)
  sum_mx_pre = sum(mx_all[:,0]) / sum(mx_all[:,2])
  sum_mx_rec = sum(mx_all[:,0]) / sum(mx_all[:,1])
  sum_mx_f1 = 2 * sum_mx_pre*sum_mx_rec / (sum_mx_rec + sum_mx_pre)
  br_valid = sum(num_valid_br)
  mx_valid = sum(num_valid_mx)
  print("Bracket only -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" 
          %(br_valid, sum_br_pre, sum_br_rec, sum_br_f1) ) 
  print("Matched XX   -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" 
          %(mx_valid, sum_mx_pre, sum_mx_rec, sum_mx_f1) ) 

  if br_valid >= 5000 and sum_br_f1 >= 80: 
    print("Very good model")
    checkpoint_path = os.path.join(FLAGS.train_dir, "good-model.ckpt")
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)


# TODO:
# FIX DECODE CODE
def decode():
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model"):
      model, steps_done = create_model(sess, True, attention=FLAGS.attention, model_path=FLAGS.model_path)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    sents_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.sents" % FLAGS.input_vocab_size)
    parse_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.parse" % FLAGS.output_vocab_size)
    sents_vocab, _ = data_utils.initialize_vocabulary(sents_vocab_path)
    _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

    start_time = time.time()
    # Decode 
    with open(FLAGS.decode_input_path, 'r') as fin, open(FLAGS.decode_output_path, 'w') as fout:
      for line in fin:
        sentence = line.strip()
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), sents_vocab)
        try:
          bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
        except:
          print("Input sentence does not fit in any buckets. Skipping... ")
          print("\t", line)
          continue
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        decoded_sentence = " ".join([tf.compat.as_str(rev_parse_vocab[output]) for output in outputs]) + '\n'
        fout.write(decoded_sentence)
    time_elapsed = time.time() - start_time
    print("Decoding time: ", time_elapsed)

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
          encoder_inputs, decoder_inputs, target_weights = model.get_ordered_batch(
                  data_set, bucket_id, bucket_offset)
          model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
          print(len(encoder_inputs), len(decoder_inputs))



def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
