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
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
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


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def basic_rnn_seq2seq(
    encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None):
  """Basic RNN sequence-to-sequence model.

  This model first runs an RNN to encode encoder_inputs into a state vector,
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell type, but don't share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
    _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
    return rnn_decoder(decoder_inputs, enc_state, cell)


def embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,
                          embedding_size, output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True, scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_projection is not None:
    proj_weights = ops.convert_to_tensor(output_projection[0],
                                         dtype=dtypes.float32)
    proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
    proj_biases = ops.convert_to_tensor(
        output_projection[1], dtype=dtypes.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
    embedding = variable_scope.get_variable("embedding",
            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
    return rnn_decoder(emb_inp, initial_state, cell,
                       loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                          num_encoder_symbols, num_decoder_symbols,
                          embedding_size, output_projection=None,
                          feed_previous=False, dtype=dtypes.float32,
                          scope=None):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
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
                    use_conv=False, conv_filter_width=10, conv_num_channels=10,
                    output_size=None, num_heads=1, loop_function=None,
                    dtype=dtypes.float32, scope=None,
                    initial_state_attention=False, attention_vec_size=None):
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

    attention_vec_size = 64#attn_size  # Size of query vectors for attention.

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

          alpha = nn_ops.softmax(s)
          alpha = array_ops.reshape(alpha, [-1, attn_length, 1, 1])
          alphas.append(alpha)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(alpha * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds, alphas

    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])

    batch_alpha_size = array_ops.pack([batch_size, attn_length, 1, 1])
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
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
          attns, alphas = attention(cell_output, alphas)
      else:
        attns, alphas = attention(cell_output, alphas)
        
      #with variable_scope.variable_scope("OutputAttnOutputProjection"):
      #  output = linear([cell_output], output_size, False) 
      #with variable_scope.variable_scope("AttnOutputProjection"):
      #  output += linear(attns, output_size, True) 
        #output = cell_output + linear(attns, output_size, True) 

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True) 
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, 
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=20, 
                                num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False):
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
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, 
        use_conv=use_conv, conv_filter_width=conv_filter_width,  ## Convolution feature param
        conv_num_channels=conv_num_channels,                    
        output_size=output_size,
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, seq_len, 
                                seq_len_frames, cell, cell_frame,
                                cell_dec, num_layers, 
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size, skip_step=1, bi_dir=False, 
                                bi_dir_frame=False,
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=20, 
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None, initial_state_attention=False):
  with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
    encoder_size = len(encoder_inputs)
    frame_rnn_final_states = []
    ## Run frame level RNN for all the words
    for w_id in xrange(encoder_size):
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), 
                reuse=True if w_id > 0 else None):
            if bi_dir_frame:
                _, state_fw, state_bw = rnn.bidirectional_rnn(cell_frame, cell_frame, \
                    encoder_inputs[w_id], sequence_length=seq_len_frames[w_id], dtype=dtype, \
                    scope="frame")
                comb_state = array_ops.concat(1, [state_fw.h, state_bw.h])
                frame_rnn_final_states.append(comb_state)
            else:
                _, state = rnn.rnn(cell_frame, encoder_inputs[w_id], \
                    sequence_length=seq_len_frames[w_id], dtype=dtype, scope="frame")
                frame_rnn_final_states.append(state.h)
    
    ## For layers above the encoder input is the final hidden state of all word-level RNNs
    encoder_inputs = frame_rnn_final_states

    if bi_dir:
      encoder_outputs, encoder_state, _ = rnn.bidirectional_rnn(cell, cell,
            encoder_inputs, sequence_length=seq_len, dtype=dtype)
    else:
      encoder_outputs, encoder_state = rnn.rnn(cell, 
            encoder_inputs, sequence_length=seq_len, dtype=dtype)

    if skip_step != 1 and num_layers != 1:
        encoder_state_list = [encoder_state]
        for layer_id in xrange(num_layers-1):
            if len(encoder_outputs) >= skip_step:
                encoder_inputs = encoder_outputs[(skip_step - 1)::skip_step]
            else:
                encoder_inputs = encoder_outputs[-1]

            ## Update the sequence length parameter as well
            den_term = tf.ones_like(seq_len) * skip_step
            ## Floor division
            seq_len = tf.to_int64(tf.floordiv(seq_len, den_term))
            
            with variable_scope.variable_scope("RNNLayer%d" %(layer_id + 1)):
                if bi_dir:
                    encoder_outputs, encoder_state, _ = rnn.bidirectional_rnn(cell, cell, 
                            encoder_inputs, sequence_length=seq_len, dtype=dtype)
                else:
                    encoder_outputs, encoder_state = rnn.rnn(cell, 
                        encoder_inputs, sequence_length=seq_len, dtype=dtype)
                encoder_state_list.append(encoder_state)

        encoder_state = tuple(encoder_state_list)

    # First calculate a concatenation of encoder outputs to put attention on.
    if not bi_dir: 
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    else:
        top_states = [array_ops.reshape(e, [-1, 1, 2 * cell.output_size])
                  for e in encoder_outputs]
    print ("Encoder output size: %d" %len(top_states))
    attention_states = array_ops.concat(1, top_states)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell_dec, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
        return embedding_attention_decoder(
          decoder_inputs, encoder_state, attention_states, cell,
          num_decoder_symbols, embedding_size, 
          use_conv=use_conv, conv_filter_width=conv_filter_width,
          conv_num_channels=conv_num_channels,
          num_heads=num_heads,
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
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
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, seq_len, 
                       seq_len_frames, buckets, seq2seq, softmax_loss_function=None,
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
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]],
                                    seq_len, seq_len_frames)

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
