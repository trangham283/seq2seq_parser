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
import seq2seq_model
from tree_utils import add_brackets, match_length, merge_sent_tree, delete_empty_constituents


tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("dropout", True, "To use dropout or not.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 256, "Size embeddings.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_epochs", 30, "max epochs")
tf.app.flags.DEFINE_string("data_dir", \
        "/g/ssli/transitory/ttmt001/seq2seq_parser/data", \
        "Data directory")
tf.app.flags.DEFINE_string("train_dir", \
        "/g/ssli/transitory/ttmt001/seq2seq_parser/models/text_only", \
        "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "checkpoint frequency, in terms of steps")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding a given file.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS


# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 4 
source_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.sents') 
target_vocab_path = os.path.join(FLAGS.data_dir, 'vocab.parse')
source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
target_vocab, _ = data_utils.initialize_vocabulary(target_vocab_path)

def load_dev_data():
    dev_data_path = os.path.join(FLAGS.data_dir, 'dev_pitch3.pickle')
    dev_set = pickle.load(open(dev_data_path))
    return dev_set

def load_train_data():
    swtrain_data_path = os.path.join(FLAGS.data_dir, 'train_pitch3.pickle')
    # debug with small data
    #swtrain_data_path = os.path.join(FLAGS.data_dir, 'dev2_pitch3.pickle')
    train_sw = pickle.load(open(swtrain_data_path))
    train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
    print(train_bucket_sizes)
    print("# of instances: %d" %(sum(train_bucket_sizes)))
    sys.stdout.flush()
    train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
    offset_lengths = [len(x) for x in train_bucket_offsets]
    tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
    all_bucks = [x for sublist in tiled_buckets for x in sublist]
    all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
    train_set = zip(all_bucks, all_offsets)
    return train_sw, train_set

# evalb paths
#evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
#prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

def create_model(session, forward_only, dropout, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      len(source_vocab), len(target_vocab), _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, dropout=dropout)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  #if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not model_path:
  if ckpt and not model_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
    print("loaded from %d done steps" %(steps_done) )
  #elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and model_path is not None:
  elif ckpt and model_path is not None:
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
  train_sw, train_set = load_train_data()
  dev_set = load_dev_data()

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, forward_only=False, dropout=True)
    with tf.variable_scope("model", reuse=True):
      model_dev = seq2seq_model.Seq2SeqModel(
      len(source_vocab), len(target_vocab), _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=True, dropout=False)

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    epoch = model.epoch 

    while epoch <= FLAGS.max_epochs:
      print("Doing epoch: ", epoch)
      sys.stdout.flush()
      np.random.shuffle(train_set)
      for bucket_id, bucket_offset in train_set:
        #print(bucket_id, bucket_offset)
        this_sample = train_sw[bucket_id][bucket_offset:bucket_offset+FLAGS.batch_size]
        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights, seq_len = model.get_batch(
                {bucket_id: this_sample}, bucket_id, bucket_offset)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, seq_len, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if model.global_step.eval() % FLAGS.steps_per_checkpoint == 0:
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
          
      print("Current step: ", current_step)
      globstep = model.global_step.eval()
      eval_batch_size = FLAGS.batch_size
      write_time = time.time()
      write_decode(model_dev, sess, dev_set, eval_batch_size, globstep, eval_now=False)
      time_elapsed = time.time() - write_time
      
      print("decode writing time: ", time_elapsed)
      sys.stdout.flush()
      model.epoch += 1
      epoch += 1
    
def write_decode(model_dev, sess, dev_set, eval_batch_size, globstep, eval_now=False):  
  # Load vocabularies.
  #sents_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.source_vocab_file)
  #parse_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.target_vocab_file)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(source_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(target_vocab_path)

  # current progress 
  stepname = str(globstep)
  gold_file_name = os.path.join(FLAGS.train_dir, 'gold-step'+ stepname +'.txt')
  print(gold_file_name)
  # file with matched brackets
  decoded_br_file_name = os.path.join(FLAGS.train_dir, 'decoded-br-step'+ stepname +'.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(FLAGS.train_dir, 'decoded-mx-step'+ stepname +'.txt')
  
  fout_gold = open(gold_file_name, 'w')
  fout_br = open(decoded_br_file_name, 'w')
  fout_mx = open(decoded_mx_file_name, 'w')

  num_dev_sents = 0
  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, eval_batch_size) 
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+eval_batch_size]
        model_dev.batch_size = len(all_examples)        
        token_ids = [x[0] for x in all_examples]
        partition = [x[2] for x in all_examples]
        speech_feats = [x[3] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        dec_ids = [[]] * len(token_ids)
        encoder_inputs, decoder_inputs, target_weights, seq_len = model_dev.get_batch(
                {bucket_id: zip(token_ids, dec_ids)}, bucket_id, batch_offset)
        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs, \
                target_weights, seq_len, bucket_id, True)
        outputs = [np.argmax(logit, axis=1) for logit in output_logits]
        to_decode = np.array(outputs).T
        num_dev_sents += to_decode.shape[0]
        num_valid = 0
        for sent_id in range(to_decode.shape[0]):
          parse = list(to_decode[sent_id, :])
          if data_utils.EOS_ID in parse:
            parse = parse[:parse.index(data_utils.EOS_ID)]
          decoded_parse = []
          for output in parse:
              if output < len(rev_parse_vocab):
                decoded_parse.append(tf.compat.as_str(rev_parse_vocab[output]))
              else:
                decoded_parse.append("_UNK") 
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
          
  # Write to file
  fout_gold.close()
  fout_br.close()
  fout_mx.close()  

  if eval_now:
    correction_types = ["Bracket only", "Matched XX"]
    corrected_files = [decoded_br_file_name, decoded_mx_file_name]

    for c_type, c_file in zip(correction_types, corrected_files):
        cmd = [evalb_path, '-p', prm_file, gold_file_name, c_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out_lines = out.split("\n")
        vv = [x for x in out_lines if "Number of Valid sentence " in x]
        s1 = float(vv[0].split()[-1])
        m_br, g_br, t_br = process_eval(out_lines, num_dev_sents)
        
        recall = float(m_br)/float(g_br)
        prec = float(m_br)/float(t_br)
        f_score = 2 * recall * prec / (recall + prec)
        
        print("%s -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(c_type, s1, prec, recall, f_score) ) 


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
