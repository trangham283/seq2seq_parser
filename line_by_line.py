from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
 
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import seq2seq_model

from parse_nn_small import read_data


_buckets = [(10, 40), (25, 85), (40, 150)]
NUM_THREADS = 1

data_dir = '/share/data/speech/Data/ttran/parser_misc/wsj-deltrace/'
sents_train, parse_train, sents_dev, parse_dev, _, _ = data_utils.prepare_wsj_data(data_dir, 45000, 128)
dev_set = read_data(sents_dev, parse_dev)
train_set = read_data(sents_train, parse_train)

sess = tf.Session()
model = seq2seq_model.Seq2SeqModel(45000, 128, _buckets, 256, 3, 512, 5, 128, 0.5, 0.99, forward_only=True, attention=False)

train_dir = '/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/'
ckpt = tf.train.get_checkpoint_state(train_dir)
model.saver.restore(sess, ckpt.model_checkpoint_path)
steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])


sents_vocab_path = os.path.join(data_dir,"vocab%d.sents" % 45000)
parse_vocab_path = os.path.join(data_dir,"vocab%d.parse" % 128)









def get_decode_batch(model, data, bucket_id):
    encoder_size, decoder_size = model.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    for sample in data[bucket_id]:
        encoder_input, decoder_input = sample
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input + [data_utils.PAD_ID] * decoder_pad_size)
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(model.batch_size)], dtype=np.int32))
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(model.batch_size)], dtype=np.int32))
        batch_weight = np.ones(model.batch_size, dtype=np.float32)
        for batch_idx in xrange(model.batch_size):
            if length_idx < decoder_size - 1: target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID: batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

