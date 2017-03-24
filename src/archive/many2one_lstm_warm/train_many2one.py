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
import subprocess
import data_utils
import many2one_model
from tree_utils import add_brackets, match_length, merge_sent_tree, delete_empty_constituents

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("dropout", True, "To use dropout or not.")
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
tf.app.flags.DEFINE_string("model_path", None, "path to model for evaluation")
tf.app.flags.DEFINE_boolean("decode", False, "Run decoding")
tf.app.flags.DEFINE_boolean("warm_start", True, "warm start from model with SWBD-text-only")
tf.app.flags.DEFINE_string("warm_path", "/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/t2p_lstm/variables-step160000.pickle", "path to model for warm start")

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

train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
offset_lengths = [len(x) for x in train_bucket_offsets]
tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
all_bucks = [x for sublist in tiled_buckets for x in sublist]
all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
train_set = zip(all_bucks, all_offsets)
np.random.shuffle(train_set)

dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
dev_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in dev_bucket_sizes]


# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

# prep eval vocabularies
sents_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.sents" % 90000)
parse_vocab_path = os.path.join(FLAGS.data_dir,"vocab%d.parse" % 128)
sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
_, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

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
  #precision = matched/test
  #recall = matched/gold
  return matched, gold, test

def map_var_names(this_var_name):
    warm_var_name = this_var_name.replace('many2one_attention_seq2seq', 'embedding_attention_seq2seq')
    warm_var_name = warm_var_name.replace('many2one_embedding_attention_decoder', 'embedding_attention_decoder')
    warm_var_name = warm_var_name.replace('many2one_attention_decoder', 'attention_decoder')
    warm_var_name = warm_var_name.replace('text_encoder/RNN', 'RNN')
    warm_var_name = warm_var_name.replace('AttnW_text:0', 'AttnW_0:0')
    warm_var_name = warm_var_name.replace('AttnV_text:0', 'AttnV_0:0')
    warm_var_name = warm_var_name.replace('Attention_text', 'Attention_0')
    return warm_var_name

def create_model(session, forward_only, dropout, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = many2one_model.manySeq2SeqModel(
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

    if FLAGS.warm_start:
        print("Warm start")
        saved_variables = pickle.load(open(FLAGS.warm_path))
        my_variables = [v for v in tf.trainable_variables()]
        for v in my_variables:
          v_warm = map_var_names(v.name)
          print(v.name)
          print(v_warm)
          print(v_warm in saved_variables)
          if v_warm in saved_variables:
            old_v = saved_variables[v_warm]
            if old_v.shape != v.get_shape(): continue
            if "AttnOutputProjection" in v.name: continue
            print("Initializing variable with warm start:", v.name)
            session.run(v.assign(old_v))

  return model, steps_done

def train():
  """Train a sequence to sequence parser."""

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, forward_only=False, dropout=True)
    print("Now create model_dev")
    with tf.variable_scope("model", reuse=True):
      model_dev = many2one_model.manySeq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.num_layers, FLAGS.embedding_size,
      FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=True, dropout=False)

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
        # Fix bug: added EOS_ID
        for s in range(this_batch_size):
          this_sample[s][1].append(data_utils.EOS_ID)

        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights, seq_len = model.get_batch(
                {bucket_id: this_sample}, bucket_id)
        encoder_inputs_list = [text_encoder_inputs, speech_encoder_inputs] 
        start_time = time.time()
        _, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs,  
                target_weights, seq_len, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1
        
        # Once in a while, we save checkpoint, print statistics, and run evals.
        #if current_step % FLAGS.steps_per_checkpoint == 0:
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
          save_time = time.time()
          checkpoint_path = os.path.join(FLAGS.train_dir, "many2one_parse.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step,write_meta_graph=False)
          step_time, loss = 0.0, 0.0
        
        if current_step > num_remaining_steps: break

      # end of one epoch, do write decodes to do evalb
      print("Current step: ", current_step)
      globstep = model.global_step.eval()
      eval_batch_size = FLAGS.batch_size
      write_time = time.time()
      write_decode(model_dev, sess, dev_set, eval_batch_size, globstep)
      time_elapsed = time.time() - write_time
      print("decode writing time: ", time_elapsed)
      sys.stdout.flush()
    

def do_evalb(model_dev, sess, dev_set, eval_batch_size):  
  gold_file_name = os.path.join(FLAGS.train_dir, 'partial.gold.txt')
  # file with matched brackets
  decoded_br_file_name = os.path.join(FLAGS.train_dir, 'partial.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(FLAGS.train_dir, 'partial.decoded.mx.txt')
  
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
        mfccs = [x[2] for x in all_examples]
        dec_ids = [[]] * len(token_ids)
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights, seq_len = model_dev.get_batch(
                {bucket_id: zip(token_ids, dec_ids, mfccs)}, bucket_id)
        _, _, output_logits = model_dev.step(sess, [text_encoder_inputs, speech_encoder_inputs], 
                decoder_inputs, target_weights, seq_len, bucket_id, True)
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

          to_write_gold = merge_sent_tree(gold_parse, sent_text) 
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
  print("Bracket only -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(br_valid, sum_br_pre, sum_br_rec, sum_br_f1) ) 
  print("Matched XX   -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(mx_valid, sum_mx_pre, sum_mx_rec, sum_mx_f1) ) 

      
def write_decode(model_dev, sess, dev_set, eval_batch_size, globstep):  
  # Load vocabularies.
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

  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, eval_batch_size) 
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+eval_batch_size]
        model_dev.batch_size = len(all_examples)        
        token_ids = [x[0] for x in all_examples]
        mfccs = [x[2] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        dec_ids = [[]] * len(token_ids)
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights, seq_len = model_dev.get_batch(
                {bucket_id: zip(token_ids, dec_ids, mfccs)}, bucket_id)
        _, _, output_logits = model_dev.step(sess, [text_encoder_inputs, speech_encoder_inputs], 
                decoder_inputs, target_weights, seq_len, bucket_id, True)
        outputs = [np.argmax(logit, axis=1) for logit in output_logits]
        to_decode = np.array(outputs).T
        num_valid = 0
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

          to_write_gold = merge_sent_tree(gold_parse, sent_text) # account for EOS
          to_write_br = merge_sent_tree(parse_br, sent_text)
          to_write_mx = merge_sent_tree(parse_mx, sent_text)

          fout_gold.write('{}\n'.format(' '.join(to_write_gold)))
          fout_br.write('{}\n'.format(' '.join(to_write_br)))
          fout_mx.write('{}\n'.format(' '.join(to_write_mx)))
          
  # Write to file
  fout_gold.close()
  fout_br.close()
  fout_mx.close()  
   

def decode(debug=True):
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
      model_dev, steps_done = create_model(sess, forward_only=True, dropout=False, model_path=FLAGS.model_path)

    if debug:
      for v in tf.all_variables(): print(v.name, v.get_shape())

    eval_batch_size = 64
    start_time = time.time()
    do_evalb(model_dev, sess, dev_set, eval_batch_size)
    time_elapsed = time.time() - start_time
    print("Batched evalb time: ", time_elapsed)

#    start_time = time.time()
#    write_decode(model_dev, sess, dev_set, eval_batch_size) 
#    time_elapsed = time.time() - start_time
#    print("Decoding all dev time: ", time_elapsed)


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()


