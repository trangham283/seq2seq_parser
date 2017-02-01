from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import rnn_cell

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

  def __init__(self, source_vocab_size, target_vocab_size, buckets, hidden_size, 
            hidden_frame_size, num_layers, embedding_size, skip_step, bi_dir, bi_dir_frame, use_conv, 
            conv_filter_width, conv_num_channels, max_gradient_norm, batch_size, learning_rate,
            learning_rate_decay_factor, optimizer, use_lstm=False, output_keep_prob=0.8,
            num_samples=512, forward_only=False, frame_rev=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.batch_size = batch_size
    self.buckets = buckets
    
    self.frame_rev = frame_rev
    self.epoch = tf.Variable(0, trainable=False)
    self.epoch_incr = self.epoch.assign(self.epoch + 1)

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
    cell_frame = rnn_cell.GRUCell(hidden_frame_size)
    if use_lstm:
        cell_frame = rnn_cell.BasicLSTMCell(hidden_frame_size, state_is_tuple=True)
    if not forward_only: 
        cell_frame = rnn_cell.DropoutWrapper(cell_frame, output_keep_prob=output_keep_prob)

    cell = rnn_cell.GRUCell(hidden_size)
    if use_lstm:
        cell = rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
    if not forward_only: 
        ## Always use the wrapper - To not use dropout just make the probability 1
        print("Dropout used !!")
        cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
    if skip_step == 1 and num_layers > 1:
        ## Not really skipping frames then
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        cell_dec = cell
    elif skip_step != 1 and num_layers > 1:
        ## Manually handle the case of multi-layers on encoder side
        cell_dec = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        ## Num layers == 1
        cell_dec = cell

    ## SHUBHAM - Additional variable seq_len passed
    def seq2seq_f(encoder_inputs, decoder_inputs, seq_len, seq_len_frames, do_decode):
        return many2one_seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs,  
          seq_len, seq_len_frames,
          cell, cell_frame, cell_dec, num_layers, 
          source_vocab_size,
          target_vocab_size,
          embedding_size,
          skip_step=skip_step, 
          bi_dir=bi_dir, 
          bi_dir_frame=bi_dir_frame, 
          use_conv=use_conv, conv_filter_width=conv_filter_width,
          conv_num_channels=conv_num_channels,
          output_projection=output_projection,
          feed_previous=do_decode)
      
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    ## SEQUENCE LENGTH OF FRAMES OF WORDS IN A BATCH
    self.seq_len_frames = []
    for i in xrange(buckets[-1][0]):
        word_level_inputs = []
        for frame_idx in xrange(data_utils.NUM_FRAMES):
            word_level_inputs.append(tf.placeholder(tf.float32, 
                shape=[None, data_utils.MFCC_LEN], name="encoder{0}{1}".format(i, frame_idx)))
        
        self.encoder_inputs.append(word_level_inputs)
        self.seq_len_frames.append(tf.placeholder(tf.int32, shape=[None], 
                                                name="seq_len_fr{0}".format(i)))

    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    ## SHUBHAM - create seq_len tensor - Placeholder might be more natural
    ## but this ensures that it matches batch_size
    _batch_size = tf.shape(self.encoder_inputs[0])[0]
    self.seq_len = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int64)) 

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    ## SHUBHAM - Calling many2one_seq2seq since I can modify that
    if forward_only:
        #print ("Forward Only !!")
        self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.seq_len, self.seq_len_frames,
          buckets, lambda w, x, y, z: seq2seq_f(w, x, y, z, True),
          softmax_loss_function=softmax_loss_function)
    else:
        #print ("Training !!")
        self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, self.seq_len, self.seq_len_frames, 
          buckets, lambda w, x, y, z: seq2seq_f(w, x, y, z, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      ## Make optimizer as a hyperparameter
      if optimizer == "momentum":
        opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
      elif optimizer == "grad_descent":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif optimizer == "adagrad":
        opt = tf.train.AdagradOptimizer(self.learning_rate)
      else:
        opt = tf.train.AdamOptimizer(self.learning_rate)

      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params, \
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, seq_len, 
            seq_len_frames, bucket_id, forward_only):
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
        for frame_idx in xrange(data_utils.NUM_FRAMES):
            input_feed[self.encoder_inputs[l][frame_idx].name] = encoder_inputs[l][frame_idx]

        ## Sequence length of words
        input_feed[self.seq_len_frames[l].name] = seq_len_frames[l]

    for l in xrange(decoder_size):
        input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        input_feed[self.target_weights[l].name] = target_weights[l]

    ## SHUBHAM - Feed seq_len as well
    input_feed[self.seq_len.name] = seq_len

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([len(decoder_inputs[0])], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
        #print ("Forward step !!")
        output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


  def get_batch(self, data, bucket_id, sample_eval=False):
    """Get sequential batch"""
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
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
    seq_len_frames = []
    for _ in xrange(encoder_size):
        seq_len_frames.append(np.zeros(this_batch_size, dtype=np.int64))

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for i, sample in enumerate(data_source):
        ## Text input, parse output, speech input
        encoder_word_list, decoder_input, encoder_input_list = sample
        encoder_input = encoder_input_list[0]
        seq_len[i] = len(encoder_input)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils._PAD_WORD] * (encoder_size - len(encoder_input))
        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)
        ## Pad the word
        padded_encoder_input = []
        for word_id, word in enumerate(encoder_input):
            num_frames = len(word)
            ## Pad the word MFCC vecs if shorter word
            if num_frames < data_utils.NUM_FRAMES:
                pad_frames = [data_utils._PAD_VEC]*(data_utils.NUM_FRAMES - num_frames)
                if self.frame_rev:
                    updated_word = list(reversed(word)) + pad_frames
                else:
                    updated_word = word + pad_frames
                seq_len_frames[word_id][i] = num_frames
            ## Chop the word in case of too long a sequence
            else:
                if self.frame_rev:
                    updated_word = list(reversed(word[:data_utils.NUM_FRAMES]))
                else:
                    updated_word = word[:data_utils.NUM_FRAMES]
                ## seq_len_frames is required by the word level RNN
                seq_len_frames[word_id][i] = data_utils.NUM_FRAMES

            padded_encoder_input.append(updated_word)

        ## Reverse the padded word vectors
        encoder_inputs.append(list(reversed(padded_encoder_input)) + encoder_pad)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    for length_idx in xrange(encoder_size):
        ## On this input the frame RNN will run - We make it time major
        frame_rnn_inputs = []
        for frame_idx in xrange(data_utils.NUM_FRAMES):
            frame_rnn_inputs.append(np.array([encoder_inputs[batch_idx][length_idx][frame_idx]
                                for batch_idx in xrange(this_batch_size)], dtype=np.float32))
        batch_encoder_inputs.append(frame_rnn_inputs)
    
    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
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

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, seq_len, seq_len_frames

