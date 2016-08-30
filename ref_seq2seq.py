# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import sys
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
#from tensorflow.python.ops import rnn
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


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
    #else:
    #    print("Embedding updated !!")
    return emb_prev
  return loop_function


def batch_gumbel_max_sample(a, max_gumbel_noise = 1.0):
    matrix_U = -1.0*tf.log(-1.0*tf.log(tf.random_uniform(tf.shape(a),
                        minval = 0.0, maxval = max_gumbel_noise)))
    return tf.argmax(tf.sub(a, matrix_U), dimension = 1)


def _sample_posterior_and_embed(embedding, output_projection=None,
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

    #splitted_prob = tf.unpack(prev)
    #print (len(splitted_prob))
    
    prev_symbol = batch_gumbel_max_sample(prev)
    #prev_symbol = tf.multinomial(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function

    
def identity_func(x):
    return x

def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                        sampling_prob, isTraining, attention_model, 
                        attention_params ,output_size=None, 
                        num_heads=1, loop_function=None, 
                        dtype=dtypes.float32, scope=None,
                        initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
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
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % a,
                                           [attention_vec_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)))
    
        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            wts = []
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = rnn_cell._linear(query, attention_vec_size, False)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    wt = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(wt, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
                    wts.append(wt)
            return wts, ds

        def local_attention_m(query, t, D=3):
            #sys.stdout.write("local_m\n")
            #sys.stdout.flush()
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            wts = []
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = rnn_cell._linear(query, attention_vec_size, False)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

                    preceding_zeros_shape = tf.pack([batch_size, max(0, t-D)])
                    preceding_zeros = tf.zeros(preceding_zeros_shape)

                    one_size = (t - max(0, t-D)) + 1 + (min(attn_length-1, t+D) - t)
                    s = tf.slice(s, [0, max(0, t-D)], [-1, one_size])
                    a = nn_ops.softmax(s)

                    succeding_zero_cols = attn_length - 1 - min(attn_length - 1, t + D)
                    succeding_zeros_shape = tf.pack([batch_size, succeding_zero_cols])
                    succeding_zeros = tf.zeros(succeding_zeros_shape)

                    a = tf.concat(1, [preceding_zeros, a, succeding_zeros])

                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
                    wts.append(a)
            return wts, ds

        ## Variables for predictive local attention
        if attention_model == "local_p":
            some_dim = cell.output_size #attention_vec_size
            pred_v = variable_scope.get_variable("pred_v", [some_dim, 1], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            pred_W = variable_scope.get_variable("pred_W", [cell.output_size, some_dim], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        def local_attention_p(query, D=3):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            wts = []
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = rnn_cell._linear(query, attention_vec_size, False)
                    z = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + z), [2, 3])
                    a = nn_ops.softmax(s)

                    inner_mult = tf.tanh(tf.matmul(query, pred_W))
                    p_center = (attn_length - 1) * tf.sigmoid(tf.matmul(inner_mult, pred_v)) 
                    #p_int_center = tf.cast(tf.round(p_center), tf.int32)
                    
                    p_left = tf.maximum(0.0, p_center - D)
                    p_left_mat = tf.tile(p_left, tf.pack([1, attn_length]))

                    p_right = tf.minimum(attn_length - 1.0, p_center + D)
                    p_right_mat = tf.tile(p_right, tf.pack([1, attn_length]))
                    
                    s = tf.tile(tf.cast(tf.range(0, attn_length), tf.float32), tf.pack([batch_size]))
                    s = tf.reshape(s, [-1, attn_length])
                    term_selection_cond = tf.logical_and(
                            tf.greater_equal(s, p_left), tf.less_equal(s, p_right))
                    
                    zero_weight = tf.zeros(tf.pack([batch_size, attn_length]))
                    a = tf.select(term_selection_cond, a, zero_weight)
                    
                    ## Normalize a
                    epsilon = tf.constant(1e-10)  ## Adding a constant to prevent division issues
                    #epsilon = 0
                    norm_vec = tf.expand_dims(tf.reduce_sum(a, 1) + epsilon, 1)
                    norm_mat = tf.tile(norm_vec, tf.pack([1, attn_length]))
                    
                    a = tf.div(a, norm_mat)

                    gaussian_term = -2 * tf.square(s - p_center) / (D * D)
                    gaussian_factor = tf.exp(gaussian_term)
                    
                    a = a * gaussian_factor
                    
                    ## POTENTIALLY RENORMALIZE a ?????????????


                    #mask = tf.expand_dims(tf.expand_dims(gaussian_factor, -1), -1)

                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
                    wts.append(a)
            return wts, ds


        outputs = []
        all_attns = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    if not isTraining:
                        inp = loop_function(prev, i)
                    else:
                        random_prob = tf.random_uniform([])
                        inp = tf.cond(tf.less(random_prob, sampling_prob), 
                                lambda: identity_func(inp), 
                                lambda: loop_function(prev, i))

            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            if attention_params.has_key("input_feeding"):
                x = rnn_cell._linear([inp] + attns, input_size, False)
            else:
                x = inp
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
                    if attention_model == "global":
                        weights, attns = attention(cell_output)
                    elif attention_model == "local_p":
                        if attention_params.has_key("D"):
                            weights, attns = local_attention_p(cell_output, attention_params['D'])
                        else:
                            weights, attns = local_attention_p(cell_output)
                    elif attention_model == "local_m":
                        if attention_params.has_key("D"):
                            weights, attns = local_attention_m(cell_output, i, attention_params['D'])
                        else:
                            weights, attns = local_attention_m(cell_output, i)

            else:
                if attention_model == "global":
                    weights, attns = attention(cell_output)
                elif attention_model == "local_p":
                    if attention_params.has_key("D"):
                        weights, attns = local_attention_p(cell_output, attention_params['D'])
                    else:
                        weights, attns = local_attention_p(cell_output)
                elif attention_model == "local_m":
                    if attention_params.has_key("D"):
                        weights, attns = local_attention_m(cell_output, i, attention_params['D'])
                    else:
                        weights, attns = local_attention_m(cell_output, i)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = rnn_cell._linear([cell_output] + attns, output_size, False)
            
            if loop_function is not None:
                prev = output
            outputs.append(output)
            all_attns.append(weights[0])

    every_attn = tf.pack(all_attns)
    return outputs, every_attn


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states, cell, 
                                num_symbols, embedding_size, sampling_prob, isTraining,
                                attention_model, attention_params,
                                num_heads=1, output_size=None, output_projection=None,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
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
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
    with ops.device("/cpu:0"):
        embedding = variable_scope.get_variable("embedding",
                                              [num_symbols, embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    if isTraining:
        loop_function = _sample_posterior_and_embed(embedding, 
                output_projection, update_embedding=True)
    else:
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous)

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, 
        sampling_prob, isTraining, attention_model, 
        attention_params, output_size=output_size,
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, seq_len, cell_fw, cell_bw, 
                                num_encoder_symbols, num_decoder_symbols,
                                input_embedding_size, output_embedding_size,
                                sampling_prob, isTraining,
                                attention_model, attention_params,
                                num_heads=1, output_projection=None,
                                dtype=dtypes.float32,
                                scope=None, initial_state_attention=False, encoder_bi=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    {input, output}_embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "embedding_attention_seq2seq") as var_scope:
    # Encoder.
    with ops.device("/cpu:0") :
        embedding_fw = variable_scope.get_variable("embedding_fw",
                            [num_encoder_symbols, input_embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    encoder_inputs_fw = [
        embedding_ops.embedding_lookup(embedding_fw, i) for i in encoder_inputs]

    encoder_outputs_fw, encoder_state_fw = rnn.rnn(
            cell_fw, encoder_inputs_fw, sequence_length=seq_len, dtype=dtype, scope="fw")

    if encoder_bi:

        with ops.device("/cpu:0") :
            embedding_bw = variable_scope.get_variable("embedding_bw",
                            [num_encoder_symbols, input_embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        encoder_inputs_bw = [
            embedding_ops.embedding_lookup(embedding_bw, i) for i in encoder_inputs]
        ## time * batch_size tensor

        packed_inputs = tf.pack(encoder_inputs_bw)
        rev_packed_inputs = tf.reverse_sequence(packed_inputs, seq_len, seq_dim=0, batch_dim=1)

        rev_inputs = tf.unpack(rev_packed_inputs)
        
        temp_outputs_bw, temp_state_bw = rnn.rnn(cell_bw, 
                    rev_inputs, sequence_length=seq_len, dtype=dtype, scope="bw")
        
        packed_outputs = tf.pack(temp_outputs_bw)
        rev_packed_outputs = tf.reverse_sequence(packed_outputs, seq_len, seq_dim=0, batch_dim=1)

        encoder_outputs_bw = tf.unpack(rev_packed_outputs)
        
        #encoder_outputs = encoder_outputs_bw
        encoder_outputs = [tf.concat(1, [fw, bw]) 
                                for fw, bw in zip(encoder_outputs_fw, encoder_outputs_bw)]
        #encoder_state = rnn_cell._linear([temp_state_bw, encoder_state_fw], cell_fw.state_size, False)
        encoder_state = temp_state_bw

        #encoder_state = encoder_state_fw
    else:
        encoder_outputs = encoder_outputs_fw
        encoder_state = encoder_state_fw

    zeroed_cell_state = []
    for encoder_state_lev in encoder_state:
        c, h = encoder_state_lev
        c_zeroed = tf.zeros_like(c)

        zeroed_cell_state.append(rnn_cell.LSTMStateTuple(c_zeroed, h))

    encoder_state = tuple(zeroed_cell_state)


    # First calculate a concatenation of encoder outputs to put attention on.
    
    if encoder_bi:
        top_states = [array_ops.reshape(e, [-1, 1, (cell_fw.output_size + cell_bw.output_size)])
                        for e in encoder_outputs]
    else:
        top_states = [array_ops.reshape(e, [-1, 1, cell_fw.output_size])
                        for e in encoder_outputs]
    attention_states = array_ops.concat(1, top_states)

    output_size = None
    if output_projection is None:
        cell = rnn_cell.OutputProjectionWrapper(cell_fw, num_decoder_symbols)
        output_size = num_decoder_symbols

    return embedding_attention_decoder(
            decoder_inputs, encoder_state, attention_states, cell,
            num_decoder_symbols, output_embedding_size, 
            sampling_prob, isTraining, attention_model,
            attention_params, num_heads=num_heads,
            output_size=output_size, output_projection=output_projection,
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
  with ops.op_scope(logits + targets + weights, name,
                    "sequence_loss_by_example"):
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
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
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
                       buckets, seq2seq, isTraining, softmax_loss_function=None,
                       per_example_loss=False, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
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
  attns = []
  with ops.op_scope(all_inputs, name, "model_with_buckets"):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        bucket_outputs, some_val = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]], 
                                    seq_len)
        if not isTraining:
            attns.append(some_val)
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(sequence_loss_by_example(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))
        else:
          losses.append(sequence_loss(
              outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
              softmax_loss_function=softmax_loss_function))

  if isTraining:
      return outputs, losses
  else:
      return outputs, attns
