# Plot attention masks based on model
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

import data_utils
import my_seq2seq_model
#import local_seq2seq_model
import subprocess
import matplotlib.pyplot as plt
from tree_utils import add_brackets, match_length, merge_sent_tree

train_dir = '/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-0801'
#data_dir = '/scratch/ttran/swbd_data' # on cluster 
data_dir = '/scratch/ttran/Datasets/swtotal_data' # on malamute
model_path = os.path.join(train_dir, 'parse_nn_small.ckpt-50000')

# model default paremeters
batch_size = 128
input_vocab_size = 90000
output_vocab_size = 128
hidden_size = 256
embedding_size = 512
num_layers = 3
max_gradient_norm = 5.0
learning_rate = 0.1
decay_factor = 0.99
attention = True
_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

# data set paths
dev_data_path = os.path.join(data_dir, 'swbd.dev.set.pickle')
dev_set = pickle.load(open(dev_data_path))

# Load vocabularies.
sents_vocab_path = os.path.join(data_dir,"vocab%d.sents" % input_vocab_size)
parse_vocab_path = os.path.join(data_dir,"vocab%d.parse" % output_vocab_size)
sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
_, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)


def create_model_default(session, forward_only, attention=attention, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  #model = local_seq2seq_model.Seq2SeqModel(
  model = my_seq2seq_model.Seq2SeqModel(
      input_vocab_size, output_vocab_size, _buckets,
      hidden_size, num_layers, embedding_size,
      max_gradient_norm, batch_size,
      learning_rate, decay_factor,
      forward_only=forward_only, attention=attention)
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


def plot_attn_mat(model_dev, all_examples):
    model_dev.batch_size = len(all_examples)        
    token_ids = [x[0] for x in all_examples]
    gold_ids = [x[1] for x in all_examples]
    dec_ids = [[]] * len(token_ids)
    encoder_inputs, decoder_inputs, target_weights = model_dev.get_decode_batch(
            {bucket_id: zip(token_ids, dec_ids)}, bucket_id)
    _, _, output_logits, attns = model_dev.step(sess, encoder_inputs, decoder_inputs,
            target_weights, bucket_id, True)
    #_, _, output_logits, attns = model_dev.step_with_attn(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
    outputs = [np.argmax(logit, axis=1) for logit in output_logits]
    to_decode = np.array(outputs).T
    sent_id = 0
    parse = list(to_decode[sent_id, :])
    parse_all = parse[:]
    if data_utils.EOS_ID in parse: parse = parse[:parse.index(data_utils.EOS_ID)]
    decoded_parse = []
    decoded_parse_all = []
    for output in parse:
        if output < len(rev_parse_vocab): decoded_parse.append(tf.compat.as_str(rev_parse_vocab[output]))
        else: decoded_parse.append("_UNK") 
    for output in parse_all:
        if output < len(rev_parse_vocab): decoded_parse_all.append(tf.compat.as_str(rev_parse_vocab[output]))
        else: decoded_parse_all.append("_UNK") 
    gold_parse = [tf.compat.as_str(rev_parse_vocab[output]) for output in gold_ids[sent_id]]
    sent_text = [tf.compat.as_str(rev_sent_vocab[output]) for output in token_ids[sent_id]]
    mat = attns[:,0,:].T
    return encoder_inputs, sent_text, gold_parse, decoded_parse, mat     
  


sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
with tf.variable_scope("model", reuse=None):
    model_dev, steps_done = create_model_default(sess, forward_only=True, 
            attention=attention, model_path=model_path)


# Mappings to original line numbers:
dd = pickle.load(open('/scratch/ttran/Datasets/swtotal_data/swbd.dev.dict.pickle'))
tree_file = '/scratch/ttran/Datasets/parses_swbd/dev.trees'
tree_lines = open(tree_file, 'r').readlines()

# timing file:
time_file = '/scratch/ttran/Datasets/parses_swbd/dev.times'
time_lines = open(time_file, 'r').readlines()

# audio
audio_dir = '/scratch/ttran/Datasets/audio_swbd/sph/'

# Pick line from evalb result file:
evalb_line = 22 
md = dd[0]+dd[1]+dd[2]
tree_lookup = dict(zip( range(len(md) + 1), md ))
gold_tree = tree_lines[tree_lookup[evalb_line - 1]].rstrip()
this_line = time_lines[tree_lookup[evalb_line - 1]]

fname, _, _, stime, etime = this_line.split()
audio_file = audio_dir + fname[:2]+'0'+fname[2:]+'.sph'
item = 'play ' + audio_file + ' trim ' + stime + ' =' + etime
#play /scratch/ttran/Datasets/audio_swbd/sph/sw04830.sph trim 280.084375  =281.534375

#  map to plotting attention mask
if evalb_line <= len(dev_set[0]):
    bucket_id = 0
    rnum = evalb_line - 1
elif evalb_line <= len(dev_set[0] + dev_set[1]):
    bucket_id = 1
    rnum = evalb_line - len(dev_set[0]) - 1
else:
    bucket_id = 2
    rnum = evalb_line - len(dev_set[1]) - len(dev_set[0]) - 1 
    
eval_batch_size = 1
bucket_size = len(dev_set[bucket_id])
offsets = np.arange(0, bucket_size, eval_batch_size) 

# for bucket 1: try 21, 37, 70, 81, 92, 110, 119
#rnum = random.choice(offsets)
#print(rnum)
#rnum=2645
#rnum = 2712 
batch_offset = offsets[rnum]
all_examples = dev_set[bucket_id][batch_offset:batch_offset + eval_batch_size]
encoder_inputs, sent_text, gold_parse, decoded_parse, mat = plot_attn_mat(model_dev, all_examples)

inputs = [tf.compat.as_str(rev_sent_vocab[x[0]]) for x in encoder_inputs]
mat2 = mat[:, :len(decoded_parse)]
plt.figure(figsize=(20,10))
plt.matshow(mat2)    
plt.xticks(range(len(decoded_parse)), decoded_parse, fontsize=6)
plt.yticks(range(len(inputs)), inputs)
plt.colorbar()
plt.xlabel('Predicted Parse')
plt.ylabel('Input sentence') 
plt.title('  '.join(gold_parse[:-1]), fontsize=7)
#plt.title(gold_tree)
plt.savefig('attn_plots/attn_pad.png')

mm = np.flipud(mat)
mm = mm[:len(sent_text), :len(decoded_parse)]
plt.figure(figsize=(20,10))
plt.matshow(mm)    
plt.xticks(range(len(decoded_parse)), decoded_parse, fontsize=6)
plt.yticks(range(len(sent_text)), sent_text)
plt.colorbar()
plt.xlabel('Predicted Parse')
plt.ylabel('Input sentence') 
#plt.title(gold_tree)
plt.title('  '.join(gold_parse[:-1]), fontsize=7)
plt.savefig('attn_plots/attn.png')


print(gold_tree)
print(this_line)
print(item)




