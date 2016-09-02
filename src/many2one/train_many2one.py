"""
Based on parse_nn_swbd.py and debug_many2one.py
Train 2-encoder 1-decoder network for parsing
Data: switchboard

"""

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

import data_utils
import many2one_wrapper


tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 512, "Size embeddings.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("input_vocab_size", 90000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("output_vocab_size", 128, "output vocabulary size.")
tf.app.flags.DEFINE_integer("max_steps", 1000, "max number of steps, in terms of batch passes.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "checkpoint frequency, in terms of steps")
tf.app.flags.DEFINE_string("model_path", "None", "path to model for evaluation")
tf.app.flags.DEFINE_boolean("train_random", False, "train random or go through whole epochs")
FLAGS = tf.app.flags.FLAGS

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
train_buckets_scale = [0.6, 0.8, 1.0]
NUM_THREADS = 1

# data set paths
swtrain_data_path = os.path.join(FLAGS.data_dir, 'sw_train_both.pickle')
train_sw = pickle.load(open(swtrain_data_path))
dev_path = os.path.join(FLAGS.data_dir, 'sw_dev_both.pickle')
dev_set = pickle.load(open(dev_path))

# shuffle data in each bucket:
np.random.shuffle(dev_set[0])
np.random.shuffle(dev_set[1])
np.random.shuffle(dev_set[2])

train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
offset_lengths = [len(x) for x in train_bucket_offsets]
tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
all_bucks = [x for sublist in tiled_buckets for x in sublist]
all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
train_set = zip(all_bucks, all_offsets)
np.random.shuffle(train_set)

def create_model(session, forward_only, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = many2one_wrapper.manySeq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
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

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, False)

    num_remaining_steps = FLAGS.max_steps - steps_done
    print("Num remaining steps: ", num_remaining_steps)
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    epoch = 0
   
    while current_step <= num_remaining_steps:
      epoch += 1
      print("Doing epoch: ", epoch)
      np.random.shuffle(train_set) 
      for bucket_id, bucket_offset in train_set:
        this_sample = train_sw[bucket_id][bucket_offset:bucket_offset+FLAGS.batch_size]
        this_batch_size = len(this_sample)
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights = model.get_decode_batch(
                {bucket_id: this_sample}, bucket_id, this_batch_size)
        encoder_inputs_list = [text_encoder_inputs, speech_encoder_inputs] 
        start_time = time.time()
        _, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs,
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
          save_time = time.time()
          checkpoint_path = os.path.join(FLAGS.train_dir, "many2one_model.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step,write_meta_graph=False)
          step_time, loss = 0.0, 0.0

          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights = model.get_mix_batch(
                dev_set[bucket_id], bucket_id, FLAGS.batch_size)
            _, eval_loss, _ = model.step(sess, [text_encoder_inputs, speech_encoder_inputs], decoder_inputs,
                                       target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          print("Do EVALB outside separately")
          sys.stdout.flush()
    
def train_random():
  """Train a sequence to sequence parser."""
  np.random.shuffle(train_sw[0])
  np.random.shuffle(train_sw[1])
  np.random.shuffle(train_sw[2])

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, False)

    num_remaining_steps = FLAGS.max_steps - steps_done
    print("Num remaining steps: ", num_remaining_steps)
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
   
    while current_step <= num_remaining_steps:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale)) 
            if train_buckets_scale[i] > random_number_01])
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights = model.get_mix_batch(
                train_sw[bucket_id], bucket_id, FLAGS.batch_size)
        encoder_inputs_list = [text_encoder_inputs, speech_encoder_inputs] 
        start_time = time.time()
        _, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs,
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
          save_time = time.time()
          checkpoint_path = os.path.join(FLAGS.train_dir, "many2one_model.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step,write_meta_graph=False)
          step_time, loss = 0.0, 0.0

          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights = model.get_mix_batch(
                dev_set[bucket_id], bucket_id, FLAGS.batch_size)
            _, eval_loss, _ = model.step(sess, [text_encoder_inputs, speech_encoder_inputs], decoder_inputs,
                                       target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          print("Do EVALB outside separately")
          sys.stdout.flush()
    

def main(_):
  if FLAGS.train_random:
    train_random()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
