import tensorflow as tf
import numpy as np
import pandas as pd

#from tensorflow.python.ops.rnn_cell import EmbeddingWrapper
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

if __name__ == '__main__':
    np.random.seed(1)
    # the size of the hidden state for the lstm (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
    size = 1
    # 2 different sequences total
    batch_size = 3
    # the maximum steps for both sequences is 10
    n_steps = 10
    # each element of the sequence has dimension of 2
    seq_width = 2

    #feature_size = 3

    embedding_classes = 6
    embedding_size = 4

    # the first input is to be stopped at 4 steps, the second at 6 steps
    e_stop = np.array([4, 6, 2])

    initializer = tf.random_uniform_initializer(-1, 1, seed=1)

    # the sequences, has n steps of maximum size
    seq_input = tf.placeholder(tf.float32, [batch_size, n_steps, 1])

    # what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
    sequence_length = tf.placeholder(tf.int32, [batch_size])

    cell = tf.nn.rnn_cell.GRUCell(embedding_size)
    initial_state = cell.zero_state(batch_size, tf.float32)

    encoder_cell = rnn_cell.EmbeddingWrapper(
        cell, embedding_classes=embedding_classes,
        embedding_size=embedding_size, initializer=initializer)

    outputs, state = rnn.dynamic_rnn(encoder_cell, seq_input, initial_state=initial_state, sequence_length=sequence_length)

    # usual crap
    iop = tf.initialize_all_variables()
    session = tf.Session()
    session.run(iop)

    inp = []
    for i in range(0, batch_size):
        var = np.random.randint(1, 6, size=(e_stop[i], 1))
        # var = np.random.rand(e_stop[i], feature_size)
        var.resize((n_steps, 1))
        inp.append(var)

    print(inp)

    feed = {sequence_length: e_stop, seq_input: inp}

    print("outputs, should be 2 things one of length 4 and other of 6")
    outs = session.run(outputs, feed_dict=feed)
    for xx in outs:
        print(xx)

    print("states, 2 things total both of size 2, which is the size of the hidden state")
    st = session.run(state, feed_dict=feed)
    print(st)
