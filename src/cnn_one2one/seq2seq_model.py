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

from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell

# from tensorflow.models.rnn.translate import data_utils
# ttmt update: use data_utils specific to my data
import data_utils
import many2one_seq2seq


class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_vocab_size, target_vocab_size, 
                input_feats, buckets, hidden_size,
                num_layers, embedding_size,
                use_conv, conv_filter_width, conv_num_channels,
                max_gradient_norm, batch_size, learning_rate,
                learning_rate_decay_factor, optimizer, 
                attn_vec_size=64,
                use_lstm=False, output_keep_prob=0.8,
                num_samples=512, forward_only=False):

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.input_feats = input_feats
    self.buckets = buckets
    self.batch_size = batch_size
    #self.epoch = 0
    self.epoch = tf.Variable(0, trainable=False)
    self.epoch_incr = self.epoch.assign(self.epoch + 1)

    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    # Create the internal multi-layer cell for our RNN.
    cell = rnn_cell.GRUCell(hidden_size)
    if use_lstm:
        cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
    if not forward_only: 
        ## Always use the wrapper - To not use dropout just make the probability 1
        print("Dropout used !!")
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
    if num_layers > 1:
        cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    ## SHUBHAM - Additional variable seq_len passed
    def seq2seq_f(encoder_inputs, decoder_inputs, seq_len, seq_len_target, do_decode):
        return many2one_seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, 
          seq_len, seq_len_target,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=embedding_size,
          use_conv=use_conv, conv_filter_width=conv_filter_width,
          conv_num_channels=conv_num_channels,
          attn_vec_size=attn_vec_size, 
          output_projection=output_projection,
          feed_previous=do_decode)
      
    # Feeds for inputs.
    self.encoder_inputs = {}
    for key in self.input_feats:
        self.encoder_inputs[key] = tf.placeholder(tf.int32, shape=[None, None],
                                                name=key+"_encoder")

    _batch_size = tf.shape(self.encoder_inputs["word"])[0]
    self.seq_len = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int64)) 
    self.seq_len_target =  tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int32))

    self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder")
    self.targets = tf.slice(self.decoder_inputs, [1, 0], [-1, -1]) 
    
    batch_major_mask = tf.sequence_mask(self.seq_len_target, dtype=tf.float32) ## B*T
    time_major_mask = tf.transpose(batch_major_mask, [1, 0]) ## T*B
    self.target_weights = tf.reshape(time_major_mask, [-1])

    if forward_only:
      self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, self.targets,
          self.target_weights, self.seq_len, self.seq_len_target, 
          lambda w, x, y, z: seq2seq_f(w, x, y, z, True),
          softmax_loss_function=softmax_loss_function)
    else:
      self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, self.targets,
          self.target_weights, self.seq_len, self.seq_len_target, 
          lambda w, x, y, z: seq2seq_f(w, x, y, z, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      ## Make optimizer as a hyperparameter
      opt = tf.train.AdamOptimizer(self.learning_rate)
      gradients = tf.gradients(self.losses, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     max_gradient_norm)
      self.gradient_norms = norm
      self.updates = opt.apply_gradients(zip(clipped_gradients, params), \
              global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

  def step(self, session, encoder_inputs, decoder_inputs, seq_len, 
            seq_len_target, forward_only):
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

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for key in self.encoder_inputs:
        input_feed[self.encoder_inputs[key].name] = encoder_inputs[key]
    input_feed[self.decoder_inputs.name] = decoder_inputs

    ## SHUBHAM - Feed seq_len as well
    input_feed[self.seq_len.name] = seq_len
    input_feed[self.seq_len_target.name] = seq_len_target

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses]  # Loss for this batch.
    else:
        output_feed = [self.losses]  # Loss for this batch.
        output_feed.append(self.outputs)

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id, sample_eval=False, do_eval=False):
    """Get sequential batch
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = {}, []

    for input_feat in self.input_feats:
        encoder_inputs[input_feat] = []
    if sample_eval:
        this_batch_size = self.batch_size
    else:
        this_batch_size = len(data[bucket_id])

    data_source = []
    if sample_eval:
        data_source = random.sample(data[bucket_id], this_batch_size)
    else:
        data_source = data[bucket_id]

    seq_len = np.zeros((this_batch_size), dtype=np.int64)
    seq_len_target = np.zeros((this_batch_size), dtype=np.int64)

    for i, sample in enumerate(data_source):
        sent_id, encoder_input, decoder_input = sample
        seq_len[i] = len(encoder_input[self.input_feats[0]]) 
        if do_eval:
            seq_len_target[i] = decoder_size
        else:
            seq_len_target[i] = len(decoder_input) ##EOS already included
        #print (len(decoder_input) + 1)
    
    ## Get maximum lengths
    max_len_source = max(seq_len)
    #print (len(data_source))
    max_len_target = 0
    if do_eval:
        max_len_target = decoder_size
    else:
        max_len_target = max(seq_len_target)

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    sent_ids = []
    for i, sample in enumerate(data_source):
      sent_id, encoder_input, decoder_input = sample
      sent_ids.append(sent_id)
      #print (sent_id)

      encoder_pad = [data_utils.PAD_ID] * (max_len_source - seq_len[i])
      for input_feat in self.input_feats:
        ## SHUBHAM - reversing just the input
        encoder_inputs[input_feat].append(list(reversed(encoder_input[input_feat])) + encoder_pad)

      decoder_pad_size = max_len_target - len(decoder_input)
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = {}
    for input_feat in self.input_feats:
        #print  (encoder_inputs[input_feat])
        batch_encoder_inputs[input_feat] = np.array(encoder_inputs[input_feat])
        #print (batch_encoder_inputs[input_feat])
        #print (batch_encoder_inputs[input_feat].shape)

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    batch_decoder_inputs = np.zeros((max_len_target + 1, this_batch_size), dtype=np.int32) 
    for length_idx in xrange(max_len_target + 1):
        for batch_idx in xrange(this_batch_size):
            batch_decoder_inputs[length_idx][batch_idx] = decoder_inputs[batch_idx][length_idx]

    return batch_encoder_inputs, batch_decoder_inputs, seq_len, seq_len_target

