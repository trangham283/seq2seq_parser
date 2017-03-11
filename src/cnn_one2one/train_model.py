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
import argparse
import operator
import data_utils
import seqcnn_model

from bunch import bunchify
from tree_utils import add_brackets, match_length, merge_sent_tree, delete_empty_constituents

# Use the following buckets: 
#_buckets = [(10, 40), (25, 85), (40, 150)]
_buckets = [(10, 40), (25, 100), (50, 200), (100, 350)]
NUM_THREADS = 4 
FLAGS = object()
# evalb paths
evalb_path = '/homes/ttmt001/transitory/seq2seq_parser/EVALB/evalb'
prm_file = '/homes/ttmt001/transitory/seq2seq_parser/EVALB/seq2seq.prm'

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

def parse_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-lr", "--learning_rate", default=1e-3, \
            type=float, help="learning rate")
    parser.add_argument("-lr_decay", "--learning_rate_decay_factor", \
            default=0.9, type=float, help="multiplicative decay factor for learning rate")
    parser.add_argument("-opt", "--optimizer", default="adam", \
            type=str, help="Optimizer")
    parser.add_argument("-bsize", "--batch_size", default=64, \
            type=int, help="Mini-batch Size")
    parser.add_argument("-esize", "--embedding_size", default= 256, \
            type=int, help="Embedding Size")

    # rnn architecture
    parser.add_argument("-text_hsize", "--text_hidden_size", default=256, \
            type=int, help="Hidden layer size of text encoder")
    parser.add_argument("-text_num_layers", "--text_num_layers", default=2, \
            type=int, help="Number of stacked layers of text encoder")
#    parser.add_argument("-speech_hsize", "--speech_hidden_size", default=256, \
#            type=int, help="Hidden layer size of speech encoder")
#    parser.add_argument("-speech_num_layers", "--speech_num_layers", default=2, \
#            type=int, help="Number of stacked layers of speech encoder")
    parser.add_argument("-parse_hsize", "--parse_hidden_size", default=256, \
            type=int, help="Hidden layer size of decoder")
    parser.add_argument("-parse_num_layers", "--parse_num_layers", default=2, \
            type=int, help="Number of stacked layers of decoder") 

    # attention architecture
    parser.add_argument("-attn_vec_size", "--attention_vector_size", \
            default=64, type=int, help="Attention vector size in the tanh(...) operation")
    parser.add_argument("-use_conv", "--use_convolution", default=True, \
            action="store_true", help="Use convolution feature in attention")
    parser.add_argument("-conv_filter", "--conv_filter_dimension", default=40, \
            type=int, help="Convolution filter width dimension")
    parser.add_argument("-conv_channel", "--conv_num_channel", default=5, \
            type=int, help="Number of channels in the convolution feature extracted")

    # cnn architecture
    parser.add_argument("-num_filters", "--num_filters", default=5, \
            type=int, help="Number of convolution filters")
    parser.add_argument("-filter_sizes", "--filter_sizes", \
            default="10-25-50", type=str, help="Convolution filter sizes")
    parser.add_argument("-fixed_word_length", "--fixed_word_length", \
            default=50, type=int, help="fixed word length for convolution")
    
    parser.add_argument("-max_gnorm", "--max_gradient_norm", default=5.0, \
            type=float, help="Maximum allowed norm of gradients")
    parser.add_argument("-sp_scale", "--speech_bucket_scale", default=1, \
            type=int, help="Scaling factor for speech encoder buckets")
    parser.add_argument("-sv_file", "--source_vocab_file", \
            default="vocab.sents", type=str, help="Vocab file for source")
    parser.add_argument("-tv_file", "--target_vocab_file", \
            default="vocab.parse", type=str, help="Vocab file for target")
    
    parser.add_argument("-data_dir", "--data_dir", \
            default="/s0/ttmt001/speech_parsing/word_level", type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir", \
            default="/s0/ttmt001/speech_parsing/models", type=str, help="Training directory")
    parser.add_argument("-ws_path", "--warm_start_path", \
            default="None", type=str, help="Warm start model path")

    parser.add_argument("-lstm", "--lstm", default=True, \
            action="store_true", help="RNN cell to use")
    parser.add_argument("-out_prob", "--output_keep_prob", \
            default=0.8, type=float, help="Output keep probability for dropout")

    parser.add_argument("-max_epochs", "--max_epochs", default=50, \
            type=int, help="Max epochs")
    parser.add_argument("-num_check", "--steps_per_checkpoint", default=100, \
            type=int, help="Number of steps before updated model is saved")
    parser.add_argument("-eval", "--eval_dev", default=False, \
            action="store_true", help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test", default=False, \
            action="store_true", help="Get test results using the last saved model")
    parser.add_argument("-run_id", "--run_id", type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)
    
    lstm_string = ""
    if  arg_dict["lstm"]: 
        lstm_string = "rnn_" + "lstm" + "_" 

    opt_string = ""
    if arg_dict['optimizer'] != "adam":
        opt_string = 'opt_' + arg_dict['optimizer'] + '_'

    conv_string = ""
    if arg_dict['use_convolution']:
        conv_string = "use_conv_"
        conv_string += "filter_dim_" + str(arg_dict['conv_filter_dimension']) + "_"
        conv_string += "num_channel_" + str(arg_dict['conv_num_channel']) + "_"
    
    train_dir = ('hsize' + '_' + str(arg_dict['text_hidden_size']) + '_' +  
                'num_layers' + '_' + str(arg_dict['text_num_layers']) + '_' +   
                'num_filters' + '_' + str(arg_dict['num_filters']) + '_' +
                'out_prob' + '_' + str(arg_dict['output_keep_prob']) + '_' + 
                'run_id' + '_' + str(arg_dict['run_id']) )
     
    arg_dict['train_dir'] = os.path.join(arg_dict['train_base_dir'], train_dir)
    arg_dict['apply_dropout'] = False

    source_vocab_path = os.path.join(arg_dict['data_dir'], arg_dict['source_vocab_file'])
    target_vocab_path = os.path.join(arg_dict['data_dir'], arg_dict['target_vocab_file'])
    source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
    target_vocab, _ = data_utils.initialize_vocabulary(target_vocab_path)
    
    arg_dict['input_vocab_size'] = len(source_vocab)
    arg_dict['output_vocab_size'] = len(target_vocab)

    if not arg_dict['test'] and not arg_dict['eval_dev']:
        arg_dict['apply_dropout'] = True
        if not os.path.exists(arg_dict['train_dir']):
            os.makedirs(arg_dict['train_dir'])
    
        ## Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    options = bunchify(arg_dict) 
    return options
    
def load_test_data():
    test_data_path = os.path.join(FLAGS.data_dir, 'test_pitch3.pickle')
    test_set = pickle.load(open(test_data_path))
    return test_set

def load_dev_data():
    dev_data_path = os.path.join(FLAGS.data_dir, 'dev_pitch3.pickle')
    dev_set = pickle.load(open(dev_data_path))
    return dev_set

def load_train_data():
    swtrain_data_path = os.path.join(FLAGS.data_dir, 'train_pitch3.pickle')
    # debug with small data
    #swtrain_data_path = os.path.join(FLAGS.data_dir, 'dev2_pitch3.pickle')
    train_sw = pickle.load(open(swtrain_data_path))
    sample = train_sw[0][0]
    feat_dim = sample[3].shape[0]
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
    return train_sw, train_set, feat_dim

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
    warm_var_name = this_var_name.replace('many2one_attention_seq2seq', \
            'embedding_attention_seq2seq')
    warm_var_name = warm_var_name.replace('many2one_embedding_attention_decoder', \
            'embedding_attention_decoder')
    warm_var_name = warm_var_name.replace('many2one_attention_decoder', 'attention_decoder')
    warm_var_name = warm_var_name.replace('text_encoder/RNN', 'RNN')
    warm_var_name = warm_var_name.replace('AttnW_text:0', 'AttnW_0:0')
    warm_var_name = warm_var_name.replace('AttnV_text:0', 'AttnV_0:0')
    warm_var_name = warm_var_name.replace('Attention_text', 'Attention_0')
    return warm_var_name

def get_model_graph(session, feat_dim, forward_only):
  filter_sizes = [int(x) for x in FLAGS.filter_sizes.strip().split('-')]
  model = seqcnn_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets,
      FLAGS.text_hidden_size, FLAGS.parse_hidden_size, 
      FLAGS.text_num_layers, FLAGS.parse_num_layers,
      filter_sizes, FLAGS.num_filters, feat_dim, FLAGS.fixed_word_length,  
      FLAGS.embedding_size, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.attention_vector_size, FLAGS.speech_bucket_scale, 
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      FLAGS.optimizer, use_lstm=FLAGS.lstm, 
      output_keep_prob=FLAGS.output_keep_prob, forward_only=forward_only
      use_conv=FLAGS.use_convolution, conv_filter_width=FLAGS.conv_filter_dimension
      conv_num_channels=FLAGS.conv_num_channel)
  return model

def create_model(session, feat_dim, forward_only, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = get_model_graph(session, feat_dim, forward_only)
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
    #session.run(tf.initialize_all_variables())
    session.run(tf.global_variables_initializer())
    steps_done = 0
    '''
    if FLAGS.warm_start_path is not "None":
        print("Warm start")
        saved_variables = pickle.load(open(FLAGS.warm_start_path))
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
    '''
  return model, steps_done


def train():
  """Train a sequence to sequence parser; with speech encoder"""
  # prepapre data
  train_sw, train_set, feat_dim = load_train_data()
  dev_set = load_dev_data()

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating model for training")
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, feat_dim, forward_only=False)
    print("Now create model_dev")
    with tf.variable_scope("model", reuse=True):
      model_dev = get_model_graph(sess, feat_dim, forward_only=True)

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
        start_time = time.time()
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, \
                target_weights, text_seq_len, speech_seq_len = model.get_batch(
                {bucket_id: this_sample}, bucket_id, bucket_offset)
        encoder_inputs_list = [text_encoder_inputs, speech_encoder_inputs]
        #print(len(text_encoder_inputs), len(speech_encoder_inputs), [x.shape for x in speech_encoder_inputs])
        _, step_loss, _ = model.step(sess, encoder_inputs_list, decoder_inputs, target_weights, text_seq_len, speech_seq_len, bucket_id, False)
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
          sys.stdout.flush()
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)

          globstep = model.global_step.eval()
          f_score_cur = write_decode(model_dev, sess, dev_set, FLAGS.batch_size, globstep, eval_now=True)
          # only save model if improves score on dev set
          if f_score_best < f_score_cur:
              f_score_best = f_score_cur
              print("Best F-Score: %.4f" % f_score_best)
              print("Saving updated model")
              sys.stdout.flush()
              checkpoint_path = os.path.join(FLAGS.train_dir, "cnn_many2one.ckpt")
              model.saver.save(sess,checkpoint_path,global_step=model.global_step,write_meta_graph=False)

          # zero timer and loss.
          step_time, loss = 0.0, 0.0
        
      #print("Current step: ", current_step)
      #globstep = model.global_step.eval()
      #eval_batch_size = FLAGS.batch_size
      # evaluate after one epoch
      #write_time = time.time()
      #f_score_cur = write_decode(model_dev, sess, dev_set, eval_batch_size, globstep, eval_now=True)
      #write_decode(model_dev, sess, dev_set, eval_batch_size, globstep, eval_now=False)
      #time_elapsed = time.time() - write_time
      #print("decode writing time: ", time_elapsed)

      sys.stdout.flush()
      sess.run(model.epoch_incr)
      epoch += 1
    
def write_decode(model_dev, sess, dev_set, eval_batch_size, globstep, eval_now=False):  
  # Load vocabularies.
  sents_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.source_vocab_file)
  parse_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.target_vocab_file)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

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
        text_encoder_inputs, speech_encoder_inputs, decoder_inputs, target_weights, text_seq_len, speech_seq_len = model_dev.get_batch(
                {bucket_id: zip(token_ids, dec_ids, partition, speech_feats)}, \
                        bucket_id, batch_offset)
        _, _, output_logits = model_dev.step(sess, [text_encoder_inputs, speech_encoder_inputs], 
                decoder_inputs, target_weights, text_seq_len, speech_seq_len, bucket_id, True)
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
          
  # Write to file
  fout_gold.close()
  fout_br.close()
  fout_mx.close()  

  if eval_now:
    f_score_mx = 0.0
    correction_types = ["Bracket only", "Matched XX"]
    corrected_files = [decoded_br_file_name, decoded_mx_file_name]

    for c_type, c_file in zip(correction_types, corrected_files):
        cmd = [evalb_path, '-p', prm_file, gold_file_name, c_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out_lines = out.split("\n")
        vv = [x for x in out_lines if "Number of Valid sentence " in x]
        if len(vv) == 0:
            return 0.0
        s1 = float(vv[0].split()[-1])
        m_br, g_br, t_br = process_eval(out_lines, num_dev_sents)
        
        try:
            recall = float(m_br)/float(g_br)
            prec = float(m_br)/float(t_br)
            f_score = 2 * recall * prec / (recall + prec)
        except ZeroDivisionError as e:
            recall, prec, f_score = 0.0, 0.0, 0.0
        
        print("%s -- Num valid sentences: %d; p: %.4f; r: %.4f; f1: %.4f" %(c_type, s1, prec, recall, f_score) ) 
        if "XX" in c_type:
            f_score_mx = f_score
  return f_score_mx


def decode(debug=True):
  """ Decode file """
  dev_set = load_dev_data()
  eval_batch_size = FLAGS.batch_size
  sample = dev_set[0][0]
  feat_dim = sample[3].shape[0]
  
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
      model_dev, steps_done = create_model(sess, feat_dim, forward_only=True)

    if debug:
      var_dict = {}
      for v in tf.global_variables(): 
          print(v.name, v.get_shape())
          var_dict[v.name] = v.eval()
      pickle_file = os.path.join(FLAGS.train_dir, 'variables-'+ str(steps_done) +'.pickle')
      pickle.dump(var_dict, open(pickle_file, 'w'))
    

    start_time = time.time()
    #write_decode(model_dev, sess, dev_set, eval_batch_size, steps_done, eval_now=True) 
    write_decode(model_dev, sess, dev_set, eval_batch_size, steps_done, eval_now=False) 
    time_elapsed = time.time() - start_time
    print("Decoding all dev time: ", time_elapsed)

def decode_test(debug=True):
  test_set = load_test_data()
  eval_batch_size = FLAGS.batch_size
  sample = test_set[0][0]
  feat_dim = sample[3].shape[0]
  
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
      model_dev, steps_done = create_model(sess, feat_dim, forward_only=True)

    if debug:
      var_dict = {}
      for v in tf.global_variables(): 
          print(v.name, v.get_shape())
          var_dict[v.name] = v.eval()
      pickle_file = os.path.join(FLAGS.train_dir, 'variables-'+ str(steps_done) +'.pickle')
      pickle.dump(var_dict, open(pickle_file, 'w'))
    

    start_time = time.time()
    write_decode(model_dev, sess, test_set, eval_batch_size, steps_done, eval_now=False) 
    time_elapsed = time.time() - start_time
    print("Decoding time: ", time_elapsed)


if __name__ == "__main__":
  FLAGS = parse_options()
  if FLAGS.eval_dev:
    decode(True)
  elif FLAGS.test:
    decode_test(True)
  else:
    train()



