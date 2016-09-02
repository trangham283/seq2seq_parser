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
import argparse
import data_utils
import seq2seq_model
import subprocess
from tree_utils import add_brackets, match_length, merge_sent_tree

pa = argparse.ArgumentParser(description='Evaluate swbd model')
pa.add_argument('num_load', help='model step number to load')
args = pa.parse_args()
num_load = args.num_load

train_dir = '/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-seqlen-0902'
#data_dir = '/scratch/ttran/swbd_data' # on cluster 
data_dir = '/scratch/ttran/Datasets/swtotal_data' # on malamute
model_path = os.path.join(train_dir, 'parse_nn_small.ckpt-' + num_load)
batch_size = 128
input_vocab_size = 90000
output_vocab_size = 128
attention = True

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

# data set paths
dev_data_path = os.path.join(data_dir, 'swbd.dev.set.pickle')

# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

dev_set = pickle.load(open(dev_data_path))


def create_model_default(session, forward_only, attention=attention, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      90000, 128, _buckets,
      256, 3, 512,
      5.0, 128,
      0.1, 0.99,
      forward_only=forward_only, attention=attention, small_def=True)
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


def do_evalb(model_dev, sess, dev_set, eval_batch_size):  
  # Load vocabularies.
  sents_vocab_path = os.path.join(data_dir,"vocab%d.sents" % input_vocab_size)
  parse_vocab_path = os.path.join(data_dir,"vocab%d.parse" % output_vocab_size)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

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
        dec_ids = [[]] * len(token_ids)
        encoder_inputs, decoder_inputs, target_weights, seq_len = model_dev.get_decode_batch(
                {bucket_id: zip(token_ids, dec_ids)}, bucket_id)
        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs,target_weights, seq_len,
                bucket_id, True)
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
  print("Bracket only -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(br_valid, sum_br_pre, sum_br_rec, sum_br_f1) ) 
  print("Matched XX   -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(mx_valid, sum_mx_pre, sum_mx_rec, sum_mx_f1) ) 


    
      
def write_decode(model_dev, sess, dev_set):  
  # Load vocabularies.
  sents_vocab_path = os.path.join(data_dir,"vocab%d.sents" % 90000)
  parse_vocab_path = os.path.join(data_dir,"vocab%d.parse" % 128)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

  gold_file_name = os.path.join(train_dir, 'debug.gold.txt')
  # file with matched brackets
  decoded_br_file_name = os.path.join(train_dir, 'debug.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(train_dir, 'debug.decoded.mx.txt')
  
  fout_gold = open(gold_file_name, 'w')
  fout_br = open(decoded_br_file_name, 'w')
  fout_mx = open(decoded_mx_file_name, 'w')

  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, batch_size) 
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+batch_size]
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
   

def decode():
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
      model_dev, steps_done = create_model_default(sess, forward_only=True, attention=attention, model_path=model_path)
    #with tf.variable_scope("model", reuse=True):
    #  model_dev = seq2seq_model.Seq2SeqModel(90000, 128, _buckets, 256, 3, 512, 5.0, 128, 0.1, 0.99, forward_only=True, attention=attention)      
    for v in tf.all_variables():
      print(v.name, v.get_shape())

#    eval_batch_size = 64
#    start_time = time.time()
#    do_evalb(model_dev, sess, dev_set, eval_batch_size)
#    time_elapsed = time.time() - start_time
#    print("Batched evalb time: ", time_elapsed)

#    start_time = time.time()
#    write_decode(model_dev, sess, dev_set) 
#    time_elapsed = time.time() - start_time
#    print("Decoding all dev time: ", time_elapsed)



if __name__ == "__main__":
  decode() 
