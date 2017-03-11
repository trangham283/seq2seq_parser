
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
from tensorflow.contrib.rnn import core_rnn_cell as rnn_cell
from tensorflow.python.ops import variable_scope
import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear


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


def attention_decoder(decoder_inputs, initial_state, attention_states, cell, 
                    seq_len_inp, seq_len, 
                    use_conv=False, conv_filter_width=10, conv_num_channels=10,
                    output_size=None, num_heads=1, loop_function=None,
                    attn_vec_size=None,
                    dtype=dtypes.float32, scope=None,
                    initial_state_attention=False, attention_vec_size=None):
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(scope or "attention_decoder"):
    batch_size = array_ops.shape(decoder_inputs)[1]  # Needed for reshaping.
    attn_length = tf.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    emb_size = decoder_inputs.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.expand_dims(attention_states, 2)
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
    
        
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
         for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])

    batch_alpha_size = array_ops.stack([batch_size, attn_length, 1, 1])
    alphas = [array_ops.zeros(batch_alpha_size, dtype=dtype)
             for _ in xrange(num_heads)]

    ## Assumes Time major arrangement
    inputs_ta = tf.TensorArray(size=400, dtype=tf.float32)
    inputs_ta = inputs_ta.unstack(decoder_inputs)
    
    attn_mask = tf.sequence_mask(tf.cast(seq_len_inp, tf.int32), dtype=tf.float32)

    def raw_loop_function(time, cell_output, state, loop_state):
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

                    #alpha = nn_ops.softmax(s)
                    alpha = nn_ops.softmax(s) * attn_mask
                    sum_vec = tf.reduce_sum(alpha, reduction_indices=[1], keep_dims=True) + 1e-12
                    norm_term = tf.tile(sum_vec, tf.stack([1, tf.shape(alpha)[1]]))
                    alpha = alpha / norm_term

                    alpha = tf.expand_dims(alpha, 2)
                    alpha = tf.expand_dims(alpha, 3)#array_ops.reshape(alpha, [-1, attn_length, 1, 1])
                    alphas.append(alpha)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(alpha * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return tuple([tuple(ds), tuple(alphas)])

        # If loop_function is set, we use it instead of decoder_inputs.
        elements_finished = (time >= seq_len)
        finished = tf.reduce_all(elements_finished)


        if cell_output is None:
            next_state = initial_state#cell.zero_state(batch_size, dtype=tf.float32)#initial_state
            output = None
            loop_state = tuple([tuple(attns), tuple(alphas)])
            next_input = inputs_ta.read(time)
        else:
            next_state = state
            loop_state = attention(cell_output, loop_state[1])
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + list(loop_state[0]), output_size, True)

            if loop_function is not None:
                simple_input = loop_function(output, time) 
                #print ("Yolo")
                #capt_time = tf.Print(time)
            else:
                simple_input = tf.cond(finished,
                    lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                    lambda: inputs_ta.read(time),                     ##if true then read input
                    )
            # Merge input and previous attentions into one vector of the right size.
            input_size = simple_input.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input")
            with variable_scope.variable_scope("InputProjection"):
                next_input = linear([simple_input] + list(loop_state[0]), input_size, True)

        return (elements_finished, next_input, next_state, output, loop_state)

  outputs, state, _ = rnn.raw_rnn(cell, raw_loop_function)
  return outputs.concat(), state


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, seq_len_inp, seq_len_target, num_symbols, embedding_size, 
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=20, attn_vec_size=64, 
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
    emb_inp = embedding_ops.embedding_lookup(embedding, decoder_inputs)
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, seq_len_inp, seq_len_target,  
        use_conv=use_conv, conv_filter_width=conv_filter_width,  ## Convolution feature param
        conv_num_channels=conv_num_channels, 
        attn_vec_size=attn_vec_size,                    
        output_size=output_size,
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, 
                                seq_len, seq_len_target,  
                                cell,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size,
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=20, attn_vec_size=64, 
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None, initial_state_attention=False):
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
        embedding_words = {}
        comb_encoder_inputs = []
        for idx, key in enumerate(sorted(num_encoder_symbols.iterkeys())): 
            ##Necessary to sort so that the order of encoder_inputs is maintained
            if key == 'word':
                embedding_words[key] = variable_scope.get_variable("embedding_words_"+key,
                        [num_encoder_symbols[key], embedding_size])
            else:
                ##Keeping a smaller footprint
                embedding_words[key] = variable_scope.get_variable("embedding_words_"+key,
                        [num_encoder_symbols[key], 50])

            cur_inputs = embedding_ops.embedding_lookup(embedding_words[key], encoder_inputs[key])
            if idx == 0:
                comb_encoder_inputs = cur_inputs
            else:
                comb_encoder_inputs = tf.concat([comb_encoder_inputs, cur_inputs], 2) 
            
        encoder_outputs, encoder_state = rnn.dynamic_rnn(
                cell, comb_encoder_inputs, sequence_length=seq_len, dtype=dtype)
    
        attention_states = encoder_outputs 

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        return embedding_attention_decoder(
            decoder_inputs, encoder_state, attention_states, cell,
            seq_len, seq_len_target,
            num_decoder_symbols, embedding_size, 
            use_conv=use_conv, conv_filter_width=conv_filter_width,
            conv_num_channels=conv_num_channels,
            attn_vec_size=attn_vec_size,
            num_heads=num_heads,
            output_size=output_size, output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention)


def sequence_loss(logits, targets, weights, seq_len,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
        flat_targets = tf.reshape(targets, [-1])
        cost = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
        mask_cost = weights * cost
        loss = tf.reshape(mask_cost, tf.shape(targets))## T*B
        cost_per_example = tf.reduce_sum(loss, reduction_indices=0) / tf.cast(seq_len, dtypes.float32)##Reduce across time steps
         
        return tf.reduce_mean(cost_per_example)


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, 
                        seq_len, seq_len_target, seq2seq, 
                        softmax_loss_function=None, per_example_loss=False, name=None):

    all_inputs = [decoder_inputs, targets, weights]
    for inp_type, enc_input in encoder_inputs.items():
        all_inputs += [enc_input]
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        with variable_scope.variable_scope(variable_scope.get_variable_scope()):
            outputs, state = seq2seq(encoder_inputs,
                                    decoder_inputs,
                                    seq_len, seq_len_target) 

            losses = sequence_loss(outputs, targets, weights, seq_len_target, \
                    softmax_loss_function=softmax_loss_function)

            return outputs, losses
