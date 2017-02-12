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

"""Library for creating sequence-to-sequence models in TensorFlow.
"""

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

def many2one_attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False, attention_vec_size=None):
  """Many-to-one attention decoder.

  Modified from attention_decoder

  attention_states = [h_states, m_states]
  initial_state = both_encoder_states = [text_encoder_state, speech_encoder_state]

  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  #if num_heads < 1:
  #  raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states[0].get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size


  with variable_scope.variable_scope(scope or "many2one_attention_decoder"):
    text_attn_length = attention_states[0].get_shape()[1].value
    text_attn_size = attention_states[0].get_shape()[2].value
    speech_attn_length = attention_states[1].get_shape()[1].value
    speech_attn_size = attention_states[1].get_shape()[2].value
    assert text_attn_size == speech_attn_size # == hidden_size
    hidden_size = text_attn_size
    text_attention_vec_size = text_attn_size  # Size of query vectors for attention.
   
    if not attention_vec_size:
      speech_attention_vec_size = hidden_size  # Size of query vectors for attention.
    else:
      speech_attention_vec_size = attention_vec_size  # Size of query vectors for attention.

    attn_vec_size = {}
    attn_vec_size['text'] = text_attention_vec_size
    attn_vec_size['speech'] = speech_attention_vec_size
    
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    text_hidden = array_ops.reshape(attention_states[0], [-1, text_attn_length, 1, text_attn_size])
    text_k = variable_scope.get_variable("AttnW_text", [1, 1, text_attn_size, text_attention_vec_size])
    text_hidden_features = nn_ops.conv2d(text_hidden, text_k, [1, 1, 1, 1], "SAME")
    text_v = variable_scope.get_variable("AttnV_text", [text_attention_vec_size])

    speech_hidden = array_ops.reshape(attention_states[1], [-1, speech_attn_length, 1, speech_attn_size])
    speech_k = variable_scope.get_variable("AttnW_speech", [1, 1, speech_attn_size, speech_attention_vec_size])
    speech_hidden_features = nn_ops.conv2d(speech_hidden, speech_k, [1, 1, 1, 1], "SAME")
    speech_v = variable_scope.get_variable("AttnV_speech", [speech_attention_vec_size])

    text_state, speech_state = initial_state
    # this is d_0, choose from text branch for now
    state = text_state

    attn_length = {}
    v = {}
    hidden_features = {}
    hidden = {}
    #print("attn_vec_size", attn_vec_size)
    attn_length['text'] = text_attn_length
    attn_length['speech'] = speech_attn_length
    #print('attn_length', attn_length)
    v['text'] = text_v
    v['speech'] = speech_v
    hidden_features['text'] = text_hidden_features
    hidden_features['speech'] = speech_hidden_features
    hidden['text'] = text_hidden
    hidden['speech'] = speech_hidden

    def attention(query, branch):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      with variable_scope.variable_scope("Attention_%s" % branch):
          y = linear(query, attn_vec_size[branch], True)
          y = array_ops.reshape(y, [-1, 1, 1, attn_vec_size[branch]])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[branch] * math_ops.tanh(hidden_features[branch] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length[branch], 1, 1]) * hidden[branch], [1, 2])
          #ds.append(array_ops.reshape(d, [-1, attn_size]))
          ds.append(array_ops.reshape(d, [-1, hidden_size]))
      return ds


    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, hidden_size]) 
    # range(2) to account for both encoders!
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype) for _ in range(2)]

    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, hidden_size])
    if initial_state_attention:
      attns = attention(initial_state[0], 'text') + attention(initial_state[1], 'speech')
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
      
      # TT modified:
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True):
          attns = attention(cell_output, 'text') + attention(cell_output, 'speech')
      else:
        attns = attention(cell_output, 'text') + attention(cell_output, 'speech')

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def many2one_embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, 
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False, attention_vec_size=None):
  """ Many-to-one embedding attention decoder.

  Based on embedding_attention_decoder originally
  attention_states = [h_states, m_states]
  initial_state = both_encoder_states = [text_encoder_state, speech_encoder_state]

  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(scope or "many2one_embedding_attention_decoder"):
    embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(embedding, output_projection, 
            update_embedding_for_previous) if feed_previous else None
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return many2one_attention_decoder(
        emb_inp, initial_state, attention_states, cell, output_size=output_size,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention, attention_vec_size=attention_vec_size)


def many2one_attention_seq2seq(encoder_inputs_list, 
        decoder_inputs, text_len, 
        text_cell, speech_cell, parse_cell,
        num_encoder_symbols, num_decoder_symbols,
        embedding_size, output_projection=None,
        feed_previous=False, dtype=dtypes.float32,
        scope=None, initial_state_attention=False,
        attention_vec_size=None):

  text_encoder_inputs, speech_encoder_inputs = encoder_inputs_list
  with variable_scope.variable_scope(scope or "many2one_attention_seq2seq"):
    with ops.device("/cpu:0"):
      embedding_words = variable_scope.get_variable("embedding_words",
              [num_encoder_symbols, embedding_size])

    text_encoder_inputs = [embedding_ops.embedding_lookup(embedding_words, i) 
            for i in text_encoder_inputs] 
    # Encoder.
    with variable_scope.variable_scope(scope or "text_encoder"):
      text_encoder_outputs, text_encoder_state = rnn.rnn(
              text_cell, text_encoder_inputs, sequence_length=text_len, dtype=dtype)

    with variable_scope.variable_scope(scope or "speech_encoder"):
      speech_encoder_outputs, speech_encoder_state = rnn.rnn(
              speech_cell, speech_encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    text_top_states = [array_ops.reshape(e, [-1, 1, text_cell.output_size])
                  for e in text_encoder_outputs]
    # h_states =  attention_states in original code
    h_states = array_ops.concat(1, text_top_states)
    
    speech_top_states = [array_ops.reshape(e, [-1, 1, speech_cell.output_size])
                  for e in speech_encoder_outputs]
    m_states = array_ops.concat(1, speech_top_states)

    attention_states = [h_states, m_states]
    both_encoder_states = [text_encoder_state, speech_encoder_state]

    # Decoder.
    output_size = None
    if output_projection is None:
      parse_cell = rnn_cell.OutputProjectionWrapper(parse_cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return many2one_embedding_attention_decoder(
          decoder_inputs, both_encoder_states, attention_states,
          parse_cell, num_decoder_symbols, embedding_size, 
          output_size=output_size, output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention, 
          attention_vec_size=attention_vec_size)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=reuse):
        outputs, state = many2one_embedding_attention_decoder(
            decoder_inputs, both_encoder_states, attention_states, 
            parse_cell, num_decoder_symbols, embedding_size, 
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

def many2one_model_with_buckets(encoder_inputs_list, decoder_inputs, targets, weights,
                       text_len, buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None, spscale=5):
  
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
  with ops.op_scope(all_inputs, name, "many2one_model_with_buckets"):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                         reuse=True if j > 0 else None):
        #bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
        x = encoder_inputs_list[0][:bucket[0]]
        #print( x )
        y = encoder_inputs_list[1][:speech_buckets[j][0]]
        bucket_outputs, _ = seq2seq([x, y], 
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

