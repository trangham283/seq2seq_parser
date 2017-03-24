"""
Based on parse_nn_swbd.py and debug_many2one.py
Train 2-encoder 1-decoder network for parsing
Data: switchboard

Modified from train_many2one.py for interactive "session"
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
from tree_utils import add_brackets, match_length, merge_sent_tree

stepnum = 160000

learning_rate = 0.1
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
batch_size = 1
hidden_size = 256
embedding_size = 512
num_layers = 3
input_vocab_size = 90000
output_vocab_size = 128

data_dir = '/share/data/speech/Data/ttran/for_batch_jobs/swbd_speech/'
train_dir = '/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-many2one-0905'
model_path = os.path.join(train_dir, 'many2one_parse.ckpt-' + str(stepnum))

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
train_buckets_scale = [0.6, 0.8, 1.0]
NUM_THREADS = 1

# data set paths
dev_path = os.path.join(data_dir, 'sw_dev_both.pickle')
dev_set = pickle.load(open(dev_path))

dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
dev_bucket_offsets = [np.arange(0, x, batch_size) for x in dev_bucket_sizes]


# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

# prep eval vocabularies
sents_vocab_path = os.path.join(data_dir,"vocab%d.sents" % 90000)
parse_vocab_path = os.path.join(data_dir,"vocab%d.parse" % 128)
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


def create_model(session, forward_only, dropout, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = many2one_model.manySeq2SeqModel(
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

sess =tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
with tf.variable_scope("model", reuse=None): 
    model_dev, steps_done = create_model(sess, forward_only=True, dropout=False, model_path=model_path)

for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, batch_size) 
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+batch_size]
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




def do_evalb(model_dev, sess, dev_set, eval_batch_size):  
  gold_file_name = os.path.join(train_dir, 'partial.gold.txt')
  # file with matched brackets
  decoded_br_file_name = os.path.join(train_dir, 'partial.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(train_dir, 'partial.decoded.mx.txt')
  
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
  gold_file_name = os.path.join(train_dir, 'gold-step'+ stepname +'.txt')
  print(gold_file_name)
  # file with matched brackets
  decoded_br_file_name = os.path.join(train_dir, 'decoded-br-step'+ stepname +'.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(train_dir, 'decoded-mx-step'+ stepname +'.txt')
  
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
      model_dev, steps_done = create_model(sess, forward_only=True, dropout=False, model_path=model_path)

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



