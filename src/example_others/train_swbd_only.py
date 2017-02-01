from __future__ import absolute_import
from __future__ import division

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import cPickle as pickle
import argparse
import operator
from bunch import bunchify

import data_utils
import seq2seq_model
import subprocess
from tree_utils import add_brackets, match_length, delete_empty_constituents, merge_sent_tree

# Use the following buckets: 
_buckets = [(10, 40), (25, 85), (40, 150)]
#_buckets = [(10, 40)]
NUM_THREADS = 1
FLAGS = object()


def parse_options():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-lr_decay", "--learning_rate_decay_factor", default=0.8, type=float, help="multiplicative decay factor for learning rate")
    parser.add_argument("-opt", "--optimizer", default="adam", type=str, help="Optimizer")
    parser.add_argument("-bsize", "--batch_size", default=128, type=int, help="Mini-batch Size")

    parser.add_argument("-esize", "--embedding_size", default= 512, type=int, help="Embedding Size")
    parser.add_argument("-hfsize", "--hidden_frame_size", default=128, type=int, help="Hidden layer size for frame level RNN")
    parser.add_argument("-hsize", "--hidden_size", default=500, type=int, help="Hidden layer size")

    parser.add_argument("-num_layers", "--num_layers", default=1, type=int, help="Number of stacked layers on encoder and decoder side")
    parser.add_argument("-max_gnorm", "--max_gradient_norm", default=5.0, type=float, help="Maximum allowed norm of gradients")

    parser.add_argument("-sv_file", "--source_vocab_file", default="vocab.sents", type=str, help="Vocab file for source")
    parser.add_argument("-tv_file", "--target_vocab_file", default="vocab.parse", type=str, help="Vocab file for target")
    
    parser.add_argument("-data_dir", "--data_dir", default="/share/data/speech/shtoshni/research/sw_parsing/data/swbd_speech_hier", type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir", default="/share/data/speech/shtoshni/research/sw_parsing/speech_2_parse/cmd_s2p/models", type=str, help="Training directory")

    parser.add_argument("-bi_dir", "--bi_dir", default=False, action="store_true", help="Make encoder bi-directional")
    parser.add_argument("-bi_dir_frame", "--bi_dir_frame", default=False, action="store_true", help="Make encoder bi-directional")
    parser.add_argument("-skip_step", "--skip_step", default=1, type=int, help="Frame skipping factor as we go up the stacked layers")

    parser.add_argument("-lstm", "--lstm", default=True, action="store_true", help="RNN cell to use")
    parser.add_argument("-out_prob", "--output_keep_prob", default=0.7, type=float, help="Output keep probability for dropout")

    ## Additional features
    parser.add_argument("-use_conv", "--use_convolution", default=False, action="store_true", help="Use convolution feature in attention")
    parser.add_argument("-conv_filter", "--conv_filter_dimension", default=5, type=int, help="Convolution filter width dimension")
    parser.add_argument("-conv_channel", "--conv_num_channel", default=20, type=int, help="Number of channels in the convolution feature extracted")

    parser.add_argument("-max_epochs", "--max_epochs", default=500, type=int, help="Max epochs")
    parser.add_argument("-num_check", "--steps_per_checkpoint", default=500, type=int, help="Number of steps before updated model is saved")
    parser.add_argument("-normalize_mfcc", "--normalize_mfcc", default=False, action="store_true", help="Use normalized mfccs")
    parser.add_argument("-frame_rev", "--frame_reversal", default=False, action="store_true", help="Reverse frames of word")
    parser.add_argument("-eval", "--eval_dev", default=False, action="store_true", help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test", default=False, action="store_true", help="Get test results using the last saved model")
    parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)
    
    lstm_string = ""
    if  arg_dict["lstm"]: 
        lstm_string = "rnn_" + "lstm" + "_" 

    opt_string = ""
    if arg_dict['optimizer'] != "adam":
        opt_string = 'opt_' + arg_dict['optimizer'] + '_'

    skip_string = ""
    if arg_dict['skip_step'] != 1:
        skip_string = "skip_" + str(arg_dict['skip_step']) + "_"

    bi_dir_string = ""
    if arg_dict['bi_dir'] != False:
        bi_dir_string = "bi_dir_"

    bi_dir_frame_string = ""
    if arg_dict['bi_dir_frame'] != False:
        bi_dir_frame_string = "bi_dir_frame_"

    frame_reversal_string = ""
    if arg_dict['frame_reversal'] != False:
        frame_reversal_string = "frev_"
    conv_string = ""
    if arg_dict['use_convolution']:
        conv_string = "use_conv_"
        conv_string += "filter_dim_" + str(arg_dict['conv_filter_dimension']) + "_"
        conv_string += "num_channel_" + str(arg_dict['conv_num_channel']) + "_"
    
    norm_string = ""
    if arg_dict['normalize_mfcc']:
        norm_string = "norm_"

    train_dir = ('lr' + '_' + str(arg_dict['learning_rate']) + '_' +  
                'bsize' + '_' + str(arg_dict['batch_size']) + '_' +   
                'esize' + '_' + str(arg_dict['embedding_size']) + '_' +  
                'hsize' + '_' + str(arg_dict['hidden_size']) + '_' +  
                'hfsize' + '_' + str(arg_dict['hidden_frame_size']) + '_' +  

                skip_string + 
                bi_dir_string +
                bi_dir_frame_string + 
                conv_string + 
                frame_reversal_string + 

                'num_layers' + '_' + str(arg_dict['num_layers']) + '_' +   
                'out_prob' + '_' + str(arg_dict['output_keep_prob']) + '_' + 
                'run_id' + '_' + str(arg_dict['run_id']) + '_' + 
                opt_string + 
                lstm_string + 
                norm_string + 
                'hier')

     
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
    
def load_dev_data():
    dev_file = 'sw_dev_both.pickle'
    if FLAGS.normalize_mfcc:
        dev_file = 'sw_dev_both_norm.pickle'
    dev_data_path = os.path.join(FLAGS.data_dir, dev_file)
    dev_set = pickle.load(open(dev_data_path))

    return dev_set


def load_train_data():
    train_file = 'sw_combined_both.pickle'
    if FLAGS.normalize_mfcc:
        train_file = 'sw_combined_both_norm.pickle'
    swtrain_data_path = os.path.join(FLAGS.data_dir, train_file)
    #if FLAGS.normalize_mfcc:
    #    train_file = 'sw_dev2_both_norm.pickle'
    #swtrain_data_path = os.path.join(FLAGS.data_dir, 'sw_dev2_both.pickle')

    train_sw = pickle.load(open(swtrain_data_path))

    train_bucket_sizes = [len(train_sw[b]) for b in xrange(len(_buckets))]
    print(train_bucket_sizes)
    print ("# of instances: %d" %(sum(train_bucket_sizes)))
    sys.stdout.flush()
    

    train_bucket_offsets = [np.arange(0, x, FLAGS.batch_size) for x in train_bucket_sizes]
    offset_lengths = [len(x) for x in train_bucket_offsets]
    tiled_buckets = [[i]*s for (i,s) in zip(range(len(_buckets)), offset_lengths)]
    all_bucks = [x for sublist in tiled_buckets for x in sublist]
    all_offsets = [x for sublist in list(train_bucket_offsets) for x in sublist]
    train_set = zip(all_bucks, all_offsets)

    return train_sw, train_set


# evalb paths
evalb_path = '/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file = '/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

def get_model_graph(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.input_vocab_size, FLAGS.output_vocab_size, _buckets, FLAGS.hidden_size, 
      FLAGS.hidden_frame_size, FLAGS.num_layers, FLAGS.embedding_size, 
      FLAGS.skip_step, FLAGS.bi_dir, FLAGS.bi_dir_frame,  
      FLAGS.use_convolution, FLAGS.conv_filter_dimension, FLAGS.conv_num_channel, 
      FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate, 
      FLAGS.learning_rate_decay_factor, FLAGS.optimizer, use_lstm=FLAGS.lstm, 
      output_keep_prob=FLAGS.output_keep_prob, forward_only=forward_only,
      frame_rev=FLAGS.frame_reversal)
  return model

def create_model(session, forward_only, model_path=None):
  """Create translation model and initialize or load parameters in session."""
  model = get_model_graph(session, forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not model_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
    steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
    print("loaded from %d done steps" %(steps_done) )
    sys.stdout.flush()
  elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and model_path is not None:
    model.saver.restore(session, model_path)
    steps_done = int(model_path.split('-')[-1])
    print("Reading model parameters from %s" % model_path)
    print("loaded from %d done steps" %(steps_done) )
    sys.stdout.flush()
  else:
    print("Created model with fresh parameters.")
    sys.stdout.flush()
    session.run(tf.initialize_all_variables())
    steps_done = 0
  return model, steps_done

def train():
  """Train a sequence to sequence parser."""

  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
    sys.stdout.flush()
    with tf.variable_scope("model", reuse=None):
      model, steps_done = create_model(sess, forward_only=False)
    with tf.variable_scope("model", reuse=True):
      model_dev = get_model_graph(sess, forward_only=True)

    # Prepare data
    print("Loading data from %s" % FLAGS.data_dir)
    train_sw, train_set = load_train_data()
    dev_set = load_dev_data()

    # This is the training loop. 
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    epoch = model.epoch.eval()
    f_score_best = write_decode(model_dev, sess, dev_set) 

    while epoch <= FLAGS.max_epochs:
      print("Epochs done: %d" %epoch)
      sys.stdout.flush()
      np.random.shuffle(train_set)
      for bucket_id, bucket_offset in train_set:
        this_sample = train_sw[bucket_id][bucket_offset:bucket_offset+FLAGS.batch_size]
        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights, seq_len, seq_len_frames = \
                model.get_batch({bucket_id: this_sample}, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, \
                                                seq_len, seq_len_frames, bucket_id, False)
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
          
          f_score_cur = write_decode(model_dev, sess, dev_set) 
          ## Early stopping - ONLY UPDATING MODEL IF BETTER PERFORMANCE ON DEV
          if f_score_best < f_score_cur:
            f_score_best = f_score_cur
            # Save model
            print("Best F-Score: %.4f" % f_score_best)
            print("Saving updated model")
            sys.stdout.flush()
            checkpoint_path = os.path.join(FLAGS.train_dir, "parse_nn_small.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
          step_time, loss = 0.0, 0.0
          
          # Run evals on development set and print their perplexity.
          for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights, seq_len, seq_len_frames = \
                    model_dev.get_batch(dev_set, bucket_id, sample_eval=True)
            _, eval_loss, _ = model_dev.step(sess, encoder_inputs, decoder_inputs, \
                                target_weights, seq_len, seq_len_frames, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          #print("Do EVALB outside separately")
          sys.stdout.flush()
        
      ## Update epoch counter
      sess.run(model.epoch_incr)
      epoch += 1


def process_eval(out_lines, this_size):
  # main stuff between outlines[3:-32]
  results = out_lines[3:-32]
  try:
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

  except AssertionError as a:
    return 0, 0, 0


def write_decode(model_dev, sess, dev_set, run_id=0):  
  # Load vocabularies.
  sents_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.source_vocab_file)
  parse_vocab_path = os.path.join(FLAGS.data_dir, FLAGS.target_vocab_file)
  sents_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sents_vocab_path)
  _, rev_parse_vocab = data_utils.initialize_vocabulary(parse_vocab_path)

  model_prefix = "h_" + str(FLAGS.hidden_size) + "_n_" + str(FLAGS.num_layers) + "_d_" + str(FLAGS.output_keep_prob) + "_"
  gold_file_name = os.path.join(FLAGS.train_dir, str(run_id) + 'debug.gold.txt')

  # file with matched brackets
  decoded_br_file_name = os.path.join(FLAGS.train_dir, model_prefix + str(run_id) + 'debug.decoded.br.txt')
  # file filler XX help as well
  decoded_mx_file_name = os.path.join(FLAGS.train_dir, model_prefix + str(run_id) + 'debug.decoded.mx.txt')
  
  fout_gold = open(gold_file_name, 'w')
  fout_br = open(decoded_br_file_name, 'w')
  fout_mx = open(decoded_mx_file_name, 'w')

  num_dev_sents = 0
  all_input = None
  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, FLAGS.batch_size) 
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+FLAGS.batch_size]
        #model_dev.batch_size = len(all_examples)        
        token_ids = [x[0] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        encoder_inputs, decoder_inputs, target_weights, seq_len, seq_len_frames = \
                model_dev.get_batch({bucket_id: all_examples}, bucket_id)
        all_input = [encoder_inputs, decoder_inputs, target_weights, seq_len, seq_len_frames]
        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs,\
                        target_weights, seq_len, seq_len_frames,  bucket_id, True)
        outputs = [np.argmax(logit, axis=1) for logit in output_logits]
        to_decode = np.array(outputs).T
        
        num_dev_sents += to_decode.shape[0]
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


def dump_trainable_vars():
    model_prefix = FLAGS.train_dir.split("/")[-1]
    model_file = os.path.join(FLAGS.train_dir, "s2p_tuned_" + model_prefix + ".pickle")
    
    with open(model_file, "w") as f:
        var_name_to_val = {}
        for var in tf.trainable_variables():
            var_name_to_val[var.name] = var.eval()

        pickle.dump(var_name_to_val, f)
   

def decode():
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model"):
      model_dev, steps_done = create_model(sess, forward_only=True)
    
    print ("Epochs done: %d" %model_dev.epoch.eval())
    dump_trainable_vars()
    dev_set = load_dev_data()

    start_time = time.time()
    write_decode(model_dev, sess, dev_set) 
    time_elapsed = time.time() - start_time
    print("Decoding all dev time: ", time_elapsed)



if __name__ == "__main__":
    FLAGS = parse_options()
    if FLAGS.test:
        self_test()
    elif FLAGS.eval_dev:
        decode()
    else:
        train()

