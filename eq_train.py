import tensorflow as tf
import numpy as np
import time
import cPickle as pickle
import many2one_seq2seq

# parameters to play with
batch_size = 10
num_steps = 2 

embedding_dim = 100
memory_dim = 30

# toy task: learn simple arithmetics
syms = ['GO', 'PAD', 'EOS', '*', '+', '=']
numset = pickle.load(open('num_set_small.pickle'))
#max_num = 500
max_num = 50
nums = numset.union(range(max_num))
words = syms + [str(n) for n in nums]
ids = range(len(words))
word2id = dict(zip(words, ids))
id2word = dict(zip(ids, words))
print len(words)

def convert_sym2array(array_of_syms):
    num_arr = []
    for row in range(array_of_syms.shape[0]):
        thisrow = [word2id[x] for x in array_of_syms[row,:]]
        num_arr.append(thisrow)
    return np.array(num_arr)

def convert_array2sym(array_of_ints):
    num_arr = []
    for row in range(array_of_ints.shape[0]):
        thisrow = [id2word[x] for x in array_of_ints[row,:]]
        num_arr.append(thisrow)
    return np.array(num_arr)


in_seq_len = 3   # addition or multiplication of two numbers
out_seq_len = 2  # equal sign and the result
vocab_size = len(word2id)

# load and convert data to int
train_data = pickle.load(open('eq_train_small.pickle'))

##
# CHOOSE SUBSET OR ALL DATA
##
rawX = train_data['X'][:1000]
rawY = train_data['Y'][:1000]
allX = convert_sym2array(rawX)
allY = convert_sym2array(rawY)


# Begin building model graph
# placeholder for encoder, decoder, weights
enc_inp = [tf.placeholder(tf.int32, shape=(batch_size, 1), name = "inp%i" % t) for t in range(in_seq_len)]
labels = [tf.placeholder(tf.int32, shape=(batch_size, 1), name = "labels%i" % t) for t in range(out_seq_len)]
dec_inp = ([tf.zeros_like(labels[0], dtype=np.int32, name="GO")] + labels[:-1])
weights = [tf.ones_like(dec_inp_t, dtype=tf.float32) for dec_inp_t in dec_inp]
input_lengths = [3, 3, 3, 2, 2, 1, 3, 3, 3, 3]

# build the graph
cell = tf.nn.rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = many2one_seq2seq.embedding_attention_dynamic_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim, input_lengths)

# standard cross entropy loss function
loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights)

# specify optimizer
learning_rate = 0.05
momentum = 0.9
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# run session
# limit num threads
NUM_THREADS = 1 
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
sess.run(tf.initialize_all_variables())

data_size = len(allX)
batch_offsets = np.arange(0, data_size, batch_size)

#logs_path = '/tmp/tensorflow_logs/example'
#summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

start_time = time.time()
loss_per_step = []
for _ in range(num_steps):
    print "Step"
    loss_per_batch = 0
    for batch_offset in batch_offsets:
        thisX = allX[batch_offset:batch_offset+batch_size, :]
        thisY = allY[batch_offset:batch_offset+batch_size, :]
        # Dimshuffle to seq_len * batch_size
        X = np.array(thisX).T
        Y = np.array(thisY).T
        feed_dict = {enc_inp[t]: X[t] for t in range(in_seq_len)}
        feed_dict.update({labels[t]: Y[t] for t in range(out_seq_len)})
        _, loss_t = sess.run([train_op, loss], feed_dict)
        loss_per_batch += loss_t
    loss_per_step.append(loss_per_batch)

time_elapsed = time.time() - start_time
saver = tf.train.Saver()
save_path = saver.save(sess, "eq_model.ckpt")

print "Training time: ", time_elapsed
print "Losses: first and last: ", loss_per_step[0], loss_per_step[-1]
print("Model saved in file: %s" % save_path)



