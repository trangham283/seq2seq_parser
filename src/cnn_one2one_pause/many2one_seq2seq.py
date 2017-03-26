from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                          num_encoder_symbols, num_decoder_symbols,
                          embedding_size, output_projection=None,
                          feed_previous=False, dtype=dtypes.float32,
                          scope=None):
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq"):
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(
          decoder_inputs, encoder_state, cell, num_decoder_symbols,
          embedding_size, output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=reuse):
        outputs, state = embedding_rnn_decoder(
            decoder_inputs, encoder_state, cell, num_decoder_symbols,
            embedding_size, output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        return outputs + [state]

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    return outputs_and_state[:-1], outputs_and_state[-1]



def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
        seq_len, use_conv=True, conv_filter_width=40, conv_num_channels=5,
        output_size=None, num_heads=1, loop_function=None,
        dtype=dtypes.float32, scope=None,
        initial_state_attention=False, attention_vec_size=64):
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(scope or "attention_decoder"):
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    if use_conv:
        F = []
        U = []

    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
      if use_conv:
        F.append(variable_scope.get_variable("AttnF_%d" % a, 
                                    [conv_filter_width, 1, 1, conv_num_channels]))   
        U.append(variable_scope.get_variable("AttnU_%d" % a, 
                                    [1, 1, conv_num_channels, attention_vec_size]))


    attn_mask = tf.sequence_mask(tf.cast(seq_len, tf.int32), attn_length, dtype=tf.float32)

    state = initial_state

    def attention(query, prev_alpha):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      alphas = []
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          if use_conv:
              conv_features = nn_ops.conv2d(prev_alpha[a], F[a], [1, 1, 1, 1], "SAME")
              feat_reshape = nn_ops.conv2d(conv_features, U[a], [1, 1, 1, 1], "SAME")

              # Attention mask is a softmax of v^T * tanh(...).
              s = math_ops.reduce_sum(
                      v[a] * math_ops.tanh(hidden_features[a] + y + feat_reshape), [2, 3])
          else:
              s = math_ops.reduce_sum(
                      v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          #print(s.get_shape(), attn_mask.get_shape())
          #alpha = nn_ops.softmax(s)
          alpha = nn_ops.softmax(s) * attn_mask
          sum_vec = tf.reduce_sum(alpha, reduction_indices=[1], keep_dims=True) + 1e-12
          norm_term = tf.tile(sum_vec, tf.stack([1, tf.shape(alpha)[1]]))
          alpha = alpha / norm_term
          alpha = tf.expand_dims(alpha, 2)
          alpha = tf.expand_dims(alpha, 3)

          #array_ops.reshape(alpha, [-1, attn_length, 1, 1])
          alphas.append(alpha)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(alpha * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return tuple(ds), tuple(alphas)
          
    outputs = []
    prev = None
    if initial_state_attention:
      attns = attention(initial_state)
    
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])
    
    batch_alpha_size = array_ops.stack([batch_size, attn_length, 1, 1])
    alphas = [array_ops.zeros(batch_alpha_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in alphas:  
      a.set_shape([None, attn_length, 1, 1])

    
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + list(attns), input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
          attns, alphas = attention(cell_output, alphas)
      else:
        attns, alphas = attention(cell_output, alphas)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + list(attns), output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, seq_len, num_symbols, embedding_size,  
                                use_conv, conv_filter_width, conv_num_channels,
                                num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False,
                                attention_vec_size=64):
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
    embedding = variable_scope.get_variable("embedding",
            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, seq_len, 
        use_conv=use_conv, conv_filter_width=conv_filter_width,
        conv_num_channels=conv_num_channels,
        output_size=output_size,
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention,
        attention_vec_size=attention_vec_size)

def multipool_attention_seq2seq(
        encoder_inputs_list, decoder_inputs, 
        seq_len, feat_dim, 
        encoder_cell, parse_cell,
        num_encoder_symbols, num_pause_symbols, num_decoder_symbols,
        embedding_size, pause_size, use_conv, conv_filter_width,
        conv_num_channels,
        attention_vec_size, 
        fixed_word_length, 
        filter_sizes, num_filters,
        output_projection=None,feed_previous=False, 
        dtype=dtypes.float32,
        scope=None, initial_state_attention=False,
        use_speech=False):

  text_encoder_inputs, speech_encoder_inputs, pause_bef, pause_aft = encoder_inputs_list
  encoder_size = len(text_encoder_inputs)
  #print(encoder_size)
  #speech_encoder_inputs is size [seq_len, batch_size, fixed_word_length, feat_dim]

  with variable_scope.variable_scope(scope or "many2one_attention_seq2seq"):
    with ops.device("/cpu:0"):
      embedding_words = variable_scope.get_variable("embedding_words",
              [num_encoder_symbols, embedding_size])

    with ops.device("/cpu:0"):
      embedding_pauses = variable_scope.get_variable("embedding_pauses",
              [num_pause_symbols, pause_size])
    
    ## We need to do the embedding beforehand so that the rnn infers the input type 
    ## to be float and doesn't cause trouble in copying state after sequence length 
    ## This issue has been fixed in 0.10 version
    ## The issue is referred here - https://github.com/tensorflow/tensorflow/issues/3322
    text_encoder_inputs = [embedding_ops.embedding_lookup(embedding_words, i) 
            for i in text_encoder_inputs] 
    pause_bef = [embedding_ops.embedding_lookup(embedding_pauses, i) 
            for i in pause_bef] 
    pause_aft = [embedding_ops.embedding_lookup(embedding_pauses, i) 
            for i in pause_aft] 
    text_encoder_inputs = [tf.concat(1, [text_encoder_inputs[i], pause_bef[i], pause_aft[i] ]) \
                            for i in range(encoder_size)]
   
    if use_speech:
        # Convolution stuff happens here for speech inputs
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            print(i, filter_size)
            #with tf.name_scope("conv-maxpool-%s" % filter_size):
            with variable_scope.variable_scope(scope or "conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, feat_dim, 1, num_filters]
                W = variable_scope.get_variable("W-%d"%i, filter_shape)
                b = variable_scope.get_variable("B-%d"%i, num_filters)
                pooled_words = []  
                for j in range(encoder_size):
                    feats = speech_encoder_inputs[j]
                    feats_conv = tf.expand_dims(feats, -1)
                    conv = tf.nn.conv2d(feats_conv, W, strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    temp_length = fixed_word_length - filter_size + 1
                    pooled = tf.nn.max_pool(h,
                            ksize=[1, max(temp_length/2, 1), 1, 1],
                            strides=[1, max(temp_length/3, 1), 1, 1],
                            padding='SAME',
                            name="pool")
                    new_shape = num_filters * pooled.get_shape()[1].value
                    pooled = tf.reshape(pooled, [-1, new_shape])
                    pooled_words.append(pooled)
                pooled_outputs.append(pooled_words)

        #num_filters_total = num_filters * len(filter_sizes)
        speech_conv_outputs = tf.unpack(tf.concat(2, pooled_outputs))
        #speech_conv_outputs = [tf.reshape(x, [-1, num_filters_total]) for x in out_seq]

        # concat text_encoder_inputs and speech_conv_outputs
        both_encoder_inputs = [tf.concat(1, [text_encoder_inputs[i], speech_conv_outputs[i]]) \
                for i in range(encoder_size)]
    else:
        both_encoder_inputs = text_encoder_inputs

    # Encoder.
    with variable_scope.variable_scope(scope or "encoder"):
      encoder_outputs, encoder_states = rnn.rnn(
              encoder_cell, both_encoder_inputs, sequence_length=seq_len, dtype=dtype)
  
#    with variable_scope.variable_scope(scope or "speech_encoder"):
#      speech_encoder_outputs, speech_encoder_state = rnn.rnn(
#              speech_cell, speech_conv_outputs, sequence_length=speech_len, dtype=dtype)


    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, encoder_cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)
    
    #speech_top_states = [array_ops.reshape(e, [-1, 1, speech_cell.output_size])
    #              for e in speech_encoder_outputs]
    #m_states = array_ops.concat(1, speech_top_states)
    #attention_states = [h_states, m_states]
    #both_encoder_states = [text_encoder_state, speech_encoder_state]

    # Decoder.
    output_size = None
    if output_projection is None:
      parse_cell = rnn_cell.OutputProjectionWrapper(parse_cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs, encoder_states, attention_states,
          parse_cell, seq_len, num_decoder_symbols, embedding_size, 
          use_conv, conv_filter_width, conv_num_channels,
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention, 
          attention_vec_size=attention_vec_size)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs, both_encoder_states, attention_states, 
            parse_cell, seq_len, num_decoder_symbols, embedding_size, 
            use_conv, conv_filter_width, conv_num_channels,
            output_size=output_size, output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention, 
            attention_vec_size=attention_vec_size)
        return outputs + [state]

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    return outputs_and_state[:-1], outputs_and_state[-1]

def maxpool_attention_seq2seq(
        encoder_inputs_list, decoder_inputs, 
        seq_len, feat_dim, 
        encoder_cell, parse_cell,
        num_encoder_symbols, num_pause_symbols, num_decoder_symbols,
        embedding_size, pause_size, use_conv, conv_filter_width,
        conv_num_channels,
        attention_vec_size, 
        fixed_word_length, 
        filter_sizes, num_filters,
        output_projection=None,feed_previous=False, 
        dtype=dtypes.float32,
        scope=None, initial_state_attention=False,
        use_speech=False):

  text_encoder_inputs, speech_encoder_inputs, pause_bef, pause_aft = encoder_inputs_list
  encoder_size = len(text_encoder_inputs)
  #print(encoder_size)
  #speech_encoder_inputs is size [seq_len, batch_size, fixed_word_length, feat_dim]

  with variable_scope.variable_scope(scope or "many2one_attention_seq2seq"):
    with ops.device("/cpu:0"):
      embedding_words = variable_scope.get_variable("embedding_words",
              [num_encoder_symbols, embedding_size])

    with ops.device("/cpu:0"):
      embedding_pauses = variable_scope.get_variable("embedding_pauses",
              [num_pause_symbols, pause_size])
    
    ## We need to do the embedding beforehand so that the rnn infers the input type 
    ## to be float and doesn't cause trouble in copying state after sequence length 
    ## This issue has been fixed in 0.10 version
    ## The issue is referred here - https://github.com/tensorflow/tensorflow/issues/3322
    text_encoder_inputs = [embedding_ops.embedding_lookup(embedding_words, i) 
            for i in text_encoder_inputs] 
    pause_bef = [embedding_ops.embedding_lookup(embedding_pauses, i) 
            for i in pause_bef] 
    pause_aft = [embedding_ops.embedding_lookup(embedding_pauses, i) 
            for i in pause_aft] 
    text_encoder_inputs = [tf.concat(1, [text_encoder_inputs[i], pause_bef[i], pause_aft[i] ]) \
                            for i in range(encoder_size)]
   
    if use_speech:
        # Convolution stuff happens here for speech inputs
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            print(i, filter_size)
            #with tf.name_scope("conv-maxpool-%s" % filter_size):
            with variable_scope.variable_scope(scope or "conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, feat_dim, 1, num_filters]
                W = variable_scope.get_variable("W-%d"%i, filter_shape)
                b = variable_scope.get_variable("B-%d"%i, num_filters)
                pooled_words = []  
                for j in range(encoder_size):
                    feats = speech_encoder_inputs[j]
                    feats_conv = tf.expand_dims(feats, -1)
                    conv = tf.nn.conv2d(feats_conv, W, strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(h,
                            ksize=[1, fixed_word_length-filter_size+1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                    pooled_words.append(pooled)
                pooled_outputs.append(pooled_words)

        num_filters_total = num_filters * len(filter_sizes)
        out_seq = tf.unpack(tf.concat(2, pooled_outputs))
        speech_conv_outputs = [tf.reshape(x, [-1, num_filters_total]) for x in out_seq]

        # concat text_encoder_inputs and speech_conv_outputs
        both_encoder_inputs = [tf.concat(1, [text_encoder_inputs[i], speech_conv_outputs[i]]) \
                for i in range(encoder_size)]
    else:
        both_encoder_inputs = text_encoder_inputs

    # Encoder.
    with variable_scope.variable_scope(scope or "encoder"):
      encoder_outputs, encoder_states = rnn.rnn(
              encoder_cell, both_encoder_inputs, sequence_length=seq_len, dtype=dtype)
  
#    with variable_scope.variable_scope(scope or "speech_encoder"):
#      speech_encoder_outputs, speech_encoder_state = rnn.rnn(
#              speech_cell, speech_conv_outputs, sequence_length=speech_len, dtype=dtype)


    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, encoder_cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)
    
    #speech_top_states = [array_ops.reshape(e, [-1, 1, speech_cell.output_size])
    #              for e in speech_encoder_outputs]
    #m_states = array_ops.concat(1, speech_top_states)
    #attention_states = [h_states, m_states]
    #both_encoder_states = [text_encoder_state, speech_encoder_state]

    # Decoder.
    output_size = None
    if output_projection is None:
      parse_cell = rnn_cell.OutputProjectionWrapper(parse_cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs, encoder_states, attention_states,
          parse_cell, seq_len, num_decoder_symbols, embedding_size,  
          use_conv, conv_filter_width, conv_num_channels,
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention, 
          attention_vec_size=attention_vec_size)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs, both_encoder_states, attention_states, 
            parse_cell, seq_len, num_decoder_symbols, embedding_size, 
            use_conv, conv_filter_width, conv_num_channels,
            output_size=output_size, output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention, 
            attention_vec_size=attention_vec_size)
        return outputs + [state]

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    return outputs_and_state[:-1], outputs_and_state[-1]

def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, seq_len, cell,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size,
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None, initial_state_attention=False):
  with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
    with ops.device("/cpu:0") :
        embedding_words = variable_scope.get_variable("embedding_words",
            [num_encoder_symbols, embedding_size])
    
    ## We need to do the embedding beforehand so that the rnn infers the input type 
    ## to be float and doesn't cause trouble in copying state after sequence length 
    ## This issue has been fixed in 0.10 version
    ## The issue is referred here - https://github.com/tensorflow/tensorflow/issues/3322
    encoder_inputs = [
        embedding_ops.embedding_lookup(embedding_words, i) for i in encoder_inputs]
    encoder_outputs, encoder_state = rnn.rnn(
            cell, encoder_inputs, sequence_length=seq_len, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs, encoder_state, attention_states, cell,
          seq_len,
          num_decoder_symbols, embedding_size, 
          use_conv, conv_filter_width, conv_num_channels, attention_vec_size,
          num_heads=num_heads,
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs, encoder_state, attention_states, cell,
            seq_len,
            num_decoder_symbols, embedding_size, 
            use_conv, conv_filter_width, conv_num_channels, attention_vec_size,
            num_heads=num_heads,
            output_size=output_size, output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        return outputs + [state]

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    return outputs_and_state[:-1], outputs_and_state[-1]

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope(name, "sequence_loss_by_example",logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  with tf.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost

def many2one_model_with_buckets(encoder_inputs_list, decoder_inputs, targets, weights,
                       text_len, speech_len, buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None, spscale=1, use_speech=False):
  
  # Modified model with buckets to accept 2 encoders

  if len(encoder_inputs_list[0]) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs_list + decoder_inputs + targets + weights
  losses = []
  outputs = []
  speech_buckets = [(x*spscale, y) for (x,y) in buckets]
  with tf.name_scope(name, "many2one_model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        #bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
        x = encoder_inputs_list[0][:bucket[0]]
        pb = encoder_inputs_list[2][:bucket[0]]
        pa = encoder_inputs_list[3][:bucket[0]]
        if use_speech:
            y = encoder_inputs_list[1][:speech_buckets[j][0]]
        else:
            y = []
        bucket_outputs, _ = seq2seq([x, y, pb, pa], 
                decoder_inputs[:bucket[1]], 
                text_len)
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, seq_len, 
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with tf.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]],
                                    seq_len)

        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  return outputs, losses
