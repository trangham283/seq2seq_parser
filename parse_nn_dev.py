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
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("decode_batch", False,
                            "Set to True for decoding a given file.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("decode_input_path", "/tmp", "path to file to be decoded")
tf.app.flags.DEFINE_string("decode_output_path", "/tmp", "path to write decoded file")
tf.app.flags.DEFINE_string("model_path", "None", "path to model for evaluation")

FLAGS = tf.app.flags.FLAGS

# Use the following buckets: 
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
  """
  data_set = [[] for _ in _buckets]
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
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


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
  # Prepare WSJ data
  print("Preparing WSJ data in %s" % FLAGS.data_dir)
  sents_train, parse_train, sents_dev, parse_dev, _, _ = data_utils.prepare_wsj_data(
      FLAGS.data_dir, FLAGS.input_vocab_size, 
      FLAGS.output_vocab_size)

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

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
    dev_set = read_data(sents_dev, parse_dev)
    train_set = read_data(sents_train, parse_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
    offset_lengths = [len(x) for x in train_bucket_offsets]

    tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
    all_bucks = [x for sublist in tiled_buckets for x in sublist]
    all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
    ordered_train_set = zip(all_bucks, all_offsets)
    shuffled_train_set = ordered_train_set[:]
    np.random.shuffle(shuffled_train_set)

    train_total_size = float(sum(train_bucket_sizes))
    print("Training data bucket sizes: ", train_bucket_sizes)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    
    num_remaining_steps = FLAGS.max_steps - steps_done
    # This is the training loop. 
    if num_remaining_steps < train_total_size/FLAGS.batch_size:
      print("Training from random batches")
      sys.stdout.flush()
      # number of steps is not enough to go through data once: -- do random as in 
      # original translate code
      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      while current_step <= num_remaining_steps:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
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
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
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
          do_evalb(model_dev, model, sess, dev_set)
          sys.stdout.flush()
    
    else: # do passes over the data until max_steps is reached 
      num_step_per_pass = train_total_size/FLAGS.batch_size
      num_passes = np.floor(num_remaining_steps/num_step_per_pass)
      num_random_steps = np.ceil(num_remaining_steps - num_passes*num_step_per_pass) 
      print("%d passes to do, and %d random steps" % (num_passes, num_random_steps))
      sys.stdout.flush()

      step_time, loss = 0.0, 0.0
      current_step = 0 
      previous_losses = []
      loss_per_pass = 0.0
    
      for _ in range(int(num_passes)):
          np.random.shuffle(shuffled_train_set)
          for bucket_id, bucket_offset in shuffled_train_set: 
            # this is one step
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_ordered_batch(train_set, \
                  bucket_id, bucket_offset)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, \
                  bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint 
            loss += step_loss / FLAGS.steps_per_checkpoint 
            current_step += 1
            loss_per_pass += loss
            
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
              model.saver.save(sess, checkpoint_path, global_step=model.global_step)
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
              do_evalb(model_dev, model, sess, dev_set)
              sys.stdout.flush()
          print("Done with current pass")
          sys.stdout.flush()
      print("Done with %d passes: average loss per pass %.2f" %(num_passes, loss_per_pass/num_passes))
      print("Now do %d more random steps" %(num_random_steps))
      sys.stdout.flush()
      
      # Now the remaining steps do randomly:
      for _ in range(int(num_random_steps)):
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_set, bucket_id)
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
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          step_time, loss = 0.0, 0.0
          
          # skip this to speed things up a bit
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
          do_evalb(model_dev, model, sess, dev_set)
          sys.stdout.flush()
      
def do_evalb(model_dev, model, sess, dev_set):  
  # Load vocabularies.
  sents_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.sents" % FLAGS.input_vocab_size)
  parse_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.parse" % FLAGS.output_vocab_size)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

  gold_file_name = os.path.join(FLAGS.train_dir, 'dev.gold.txt')
  # file with matched brackets
  decoded_br_file_name = os.path.join(FLAGS.train_dir, 'dev.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(FLAGS.train_dir, 'dev.decoded.mx.txt')
  
  fout_gold = open(gold_file_name, 'w')
  fout_br = open(decoded_br_file_name, 'w')
  fout_mx = open(decoded_mx_file_name, 'w')
  
  print("Doing evalb")
  print("Debug - step: ", model.global_step.eval())
  for bucket_id in xrange(len(_buckets)):
        model_dev.batch_size = len(dev_set[bucket_id])
        all_examples = dev_set[bucket_id]
        token_ids = [x[0] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        dec_ids = [[]] * len(token_ids)
        encoder_inputs, decoder_inputs, target_weights = model_dev.get_decode_batch(
                {bucket_id: zip(token_ids, dec_ids)}, bucket_id)
        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
        outputs = [np.argmax(logit, axis=1) for logit in output_logits]
        to_decode = np.array(outputs).T
        num_valid = 0
        for sent_id in range(to_decode.shape[0]):
          parse = list(to_decode[sent_id, :])
          if data_utils.EOS_ID in parse:
            parse = parse[:parse.index(data_utils.EOS_ID)]
          # raw decoded parse
          decoded_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in parse]
          # add brackets for tree balance
          parse_br, valid = add_brackets(decoded_parse)
          num_valid += valid
          # get gold parse, gold sentence
          gold_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in gold_ids[sent_id]]
          sent_text = [tf.compat.as_str(rev_sent_vocab[output]) for output in token_ids[sent_id]]
          # parse with also matching "XX" length
          parse_mx = match_length(parse_br, sent_text)

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
  
  evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
  prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'
  cmd = [evalb_path, '-p', prm_file, gold_file_name, decoded_br_file_name]
  p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = p.communicate()
  print("Eval with completed brackets only")
  out_lines = out.split("\n")
  print([x for x in out_lines if "Number of Valid sentence " in x])
  print([x for x in out_lines if "Bracketing FMeasure " in x])
  vv = [x for x in out_lines if "Number of Valid sentence " in x] 
  ff = [x for x in out_lines if "Bracketing FMeasure " in x]
  s1 = float(vv[0].split()[-1])
  f1 = float(ff[0].split()[-1])
  
  if s1 >= 1500 and f1 >= 75: 
      print("Very good model")
      checkpoint_path = os.path.join(FLAGS.train_dir, "good-model.ckpt")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)

  cmd = [evalb_path, '-p', prm_file, gold_file_name, decoded_mx_file_name]
  p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  out, err = p.communicate()
  print("Eval with matched XXs and brackets")
  out_lines = out.split("\n")
  print([x for x in out_lines if "Number of Valid sentence " in x])
  print([x for x in out_lines if "Bracketing FMeasure " in x])



def decode():
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    model, steps_done = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    sents_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.sents" % FLAGS.input_vocab_size)
    parse_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.parse" % FLAGS.output_vocab_size)
    sents_vocab, _ = data_utils.initialize_vocabulary(sents_vocab_path)
    _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), sents_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_parse_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def decode_batch():
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
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
          encoder_inputs, decoder_inputs, target_weights = model.get_ordered_batch(data_set, bucket_id, bucket_offset)
          model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
          print(len(encoder_inputs), len(decoder_inputs))


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  elif FLAGS.decode_batch:
    decode_batch()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
