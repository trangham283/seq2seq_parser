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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import data_utils
import many2one_seq2seq

#mfcc_num = 3
#attn_vec_size = None # i.e. = hidden_size
#attn_vec_size = 64

class manySeq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, 
          text_hidden_size, speech_hidden_size, parse_hidden_size,
          text_num_layers, speech_num_layers, parse_num_layers,
          filter_sizes, num_filters, feat_dim, fixed_word_length, 
          embedding_size, max_gradient_norm, batch_size, 
          attn_vec_size, spscale,  
          learning_rate, learning_rate_decay_factor, 
          optimizer, use_lstm=True, output_keep_prob=0.8,
          num_samples=512, forward_only=False):
    """Create the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.spscale = spscale
    self.epoch = 0
    self.feat_dim = feat_dim

    self.filter_sizes = filter_sizes
    self.num_filters = num_filters

    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w = tf.get_variable("proj_w", [hidden_size, self.target_vocab_size])
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                self.target_vocab_size)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    def create_cell(hidden_size, num_layers):
        single_cell = rnn_cell.GRUCell(hidden_size)
        if use_lstm:
            print("Using LSTM")
            single_cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
            #single_cell = rnn_cell.BasicLSTMCell(hidden_size)
        if not forward_only:
            # always use dropout; set keep_prob=1 if not dropout
            print("Training mode; dropout used!")
            single_cell = rnn_cell.DropoutWrapper(single_cell, 
                    output_keep_prob=output_keep_prob)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
            #cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        return cell

    text_cell = create_cell(text_hidden_size, text_num_layers)
    speech_cell = create_cell(speech_hidden_size, speech_num_layers)
    parse_cell = create_cell(parse_hidden_size, parse_num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs_list, decoder_inputs, text_len, speech_len, do_decode, attn_vec_size):
      return many2one_seq2seq.many2one_attention_seq2seq(
          encoder_inputs_list, decoder_inputs, 
          text_len, speech_len, feat_dim,  
          text_cell, speech_cell, parse_cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=embedding_size,
          attention_vec_size=attn_vec_size,
          fixed_word_length=fixed_word_length,
          filter_sizes=filter_sizes, 
          num_filters=num_filters,
          output_projection=output_projection,
          feed_previous=do_decode)
      
    # Feeds for inputs.
    #self.encoder_inputs = []
    self.text_encoder_inputs = []
    self.speech_encoder_inputs = []
    self.speech_partitions = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.text_encoder_inputs.append(tf.placeholder(tf.int32, 
          shape=[None],name="text_encoder{0}".format(i)))
    for i in xrange(buckets[-1][0]*self.spscale):
      self.speech_encoder_inputs.append(tf.placeholder(tf.float32, 
          shape=[None, fixed_word_length, feat_dim],name="speech_encoder{0}".format(i)))
    for i in xrange(buckets[-1][1]+1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))
    self.encoder_inputs_list = [self.text_encoder_inputs, self.speech_encoder_inputs]

    # seq_len stuff:
    _batch_size = tf.shape(self.text_encoder_inputs[0])[0]
    # the constant "2" is just a placeholder
    self.text_seq_len = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int64))
    self.speech_seq_len = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int64))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs)-1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = many2one_seq2seq.many2one_model_with_buckets(
          self.encoder_inputs_list, self.decoder_inputs, targets,
          self.target_weights, self.text_seq_len, self.speech_seq_len, buckets, 
          lambda x, y, z, w: seq2seq_f(x, y, z, w, True, attn_vec_size),
          softmax_loss_function=softmax_loss_function, spscale=self.spscale)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = many2one_seq2seq.many2one_model_with_buckets(
          self.encoder_inputs_list, self.decoder_inputs, targets,
          self.target_weights, self.text_seq_len, self.speech_seq_len, buckets,
          lambda x, y, z, w: seq2seq_f(x, y, z, w, False, attn_vec_size),
          softmax_loss_function=softmax_loss_function, spscale=self.spscale)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      #opt = tf.train.AdagradOptimizer(self.learning_rate) 
      ## Make optimizer a hyperparameter
      if optimizer == "momentum":
        opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
      elif optimizer == "grad_descent":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif optimizer == "adagrad":
        print("Using adagrad optimizer")
        opt = tf.train.AdagradOptimizer(self.learning_rate)
      else:
        print("Using Adam optimizer")
        opt = tf.train.AdamOptimizer(self.learning_rate)

      for b in xrange(len(buckets)):
      # add gradient aggregration trick for less memory
        gradients = tf.gradients(self.losses[b], params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs_list, decoder_inputs, target_weights,
           text_len, speech_len, bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs_list[0]) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs_list[0]), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    speech_feats_len = encoder_inputs_list[1].shape[1]
    for l in xrange(encoder_size):
      input_feed[self.text_encoder_inputs[l].name] = encoder_inputs_list[0][l]
    #for l in xrange(encoder_size*self.spscale):
    #  input_feed[self.speech_encoder_inputs[l].name] = encoder_inputs_list[1][l] 
    
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    
    input_feed[self.speech_feats.name] = encoder_input_list[1]
    input_feed[self.speech_partitions.name] = encoder_input_list[2]
    input_feed[self.text_seq_len.name] = text_len
    input_feed[self.speech_seq_len.name] = speech_len

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([len(decoder_inputs[0])], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  
  def get_batch(self, data, bucket_id):
    """Get batches
    
    """
    this_batch_size = len(data[bucket_id])
    encoder_size, decoder_size = self.buckets[bucket_id]
    text_encoder_inputs, speech_encoder_inputs, decoder_inputs = [], [], []
    sequence_lengths = []

    for sample in data[bucket_id]:
      text_encoder_input, decoder_input, partition, speech_encoder_input = sample
      sequence_lengths.append(len(text_encoder_input))
      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(text_encoder_input))
      text_encoder_inputs.append(list(reversed(text_encoder_input)) + encoder_pad)
      # need to process speech frames for each word first
      speech_frames = []
      for frame_idx in partition:
          center_frame = int((frame_idx[0] + frame_idx[1])/2)
          start_idx = center_frame-fixed_word_length/2
          end_idx = center_frame+fixed_word_length/2
          this_word_frames = speech_encoder_input[:, max(0,start_idx):end_idx]
          if start_idx < 0 and this_word_frames.shape[1]<fixed_word_length:
              this_word_frames = np.hstack([np.zeros((self.feat_dim,-start_idx)),this_word_frames])
          if end_idx > frame_idx[1] and this_word_frames.shape[1]<fixed_word_length:
              this_word_frames = np.hstack([this_word_frames,np.zeros((self.feat_dim, end_idx-frame_idx[1]))])
          speech_frames.append(this_word_frames)
      mfcc_pad_num = encoder_size - len(text_encoder_input)
      mfcc_pad = [np.zeros((self.feat_dim, fixed_word_length)) for _ in range(mfcc_pad_num)]
      speech_stuff = list(reversed(speech_frames)) + mfcc_pad
      speech_encoder_inputs.append(speech_stuff)

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_text_encoder_inputs, batch_speech_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_text_encoder_inputs.append(
          np.array([text_encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(this_batch_size)], dtype=np.int32))

    for length_idx in xrange(encoder_size):
      current_word_feats = []
      for batch_idx in xrange(batch_size):
        current_feats = speech_encoder_inputs[batch_idx][length_idx].T
        current_feats = list(current_feats)
        current_feats = [list(x) for x in current_feats]
        current_word_feats.append(current_feats)
      batch_speech_encoder_inputs.append(current_word_feats)

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(this_batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(this_batch_size, dtype=np.float32)
      for batch_idx in xrange(this_batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64)
    return batch_text_encoder_inputs, batch_speech_encoder_inputs, \
            batch_decoder_inputs, batch_weights, sequence_lengths, \
            sequence_lengths


