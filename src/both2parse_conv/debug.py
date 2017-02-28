
'''
import cPickle as pickle
import numpy as np
f = '/s0/ttmt001/speech_parsing/word_level/dev2_pitch3.pickle'
d = pickle.load(open(f))
#this_sample = d[0][288:288+16]
this_sample = d[1][1072:1072+16]
from debug import *
all_stuff = get_batch(this_sample, batch_err_idx, seq_err_idx)

'''




import numpy as np

def get_batch(data_sample, batch_err_idx=None, seq_err_idx=None):
    """Get batches
    
    """
    PAD_ID = 0
    GO_ID = 1
    feat_dim = 3
    fixed_word_length = 50
    this_batch_size = len(data_sample)
    encoder_size, decoder_size = (25, 85)
    text_encoder_inputs, speech_encoder_inputs, decoder_inputs = [], [], []
    sequence_lengths = []

    for sample in data_sample:
      text_encoder_input, decoder_input, partition, speech_encoder_input = sample
      #print partition
      sequence_lengths.append(len(text_encoder_input))
      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(text_encoder_input))
      text_encoder_inputs.append(list(reversed(text_encoder_input)) + encoder_pad)
      # need to process speech frames for each word first
      speech_frames = []
      #fixed_word_length = fixed_word_length
      for frame_idx in partition:
          center_frame = int((frame_idx[0] + frame_idx[1])/2)
          start_idx = center_frame - int(fixed_word_length/2)
          end_idx = center_frame + int(fixed_word_length/2)
          this_word_frames = speech_encoder_input[:, max(0,start_idx):end_idx]
          if this_word_frames.shape[1]==0:  # make random if no frame info
              this_word_frames = np.random.random((feat_dim, fixed_word_length))
              print frame_idx, speech_encoder_input.shape 
          #print this_word_frames.shape[1]
          if start_idx < 0 and this_word_frames.shape[1]<fixed_word_length:
              this_word_frames = np.hstack([np.zeros((feat_dim,-start_idx)),this_word_frames])
          if end_idx > frame_idx[1] and this_word_frames.shape[1]<fixed_word_length:
              num_more = fixed_word_length - this_word_frames.shape[1]
              this_word_frames = np.hstack([this_word_frames,np.zeros((feat_dim, num_more))])
          speech_frames.append(this_word_frames)
          #print this_word_frames.shape[1]
      mfcc_pad_num = encoder_size - len(text_encoder_input)
      mfcc_pad = [np.zeros((feat_dim, fixed_word_length)) for _ in range(mfcc_pad_num)]
      speech_stuff = list(reversed(speech_frames)) + mfcc_pad
      speech_encoder_inputs.append(speech_stuff)

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_text_encoder_inputs, batch_speech_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_text_encoder_inputs.append(
          np.array([text_encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(this_batch_size)], dtype=np.int32))

    for length_idx in xrange(encoder_size):
      current_word_feats = []
      for batch_idx in xrange(this_batch_size):
        current_feats = speech_encoder_inputs[batch_idx][length_idx].T
        #print length_idx,batch_idx, current_feats.shape
        #current_feats = list(current_feats)
        #current_feats = [list(x) for x in current_feats]
        current_word_feats.append(current_feats)
        if batch_err_idx: 
            if batch_idx == batch_err_idx:
                print current_feats.shape, length_idx, batch_idx
                print 
      try:
        batch_speech_encoder_inputs.append(np.array(current_word_feats,dtype=np.float32))
        #print np.array(current_word_feats,dtype=np.float32).shape
      except:
        print "Exception"
        #print [x.shape for x in current_word_feats]

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(this_batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(this_batch_size, dtype=np.float32)
      for batch_idx in xrange(this_batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64)
    return batch_text_encoder_inputs, batch_speech_encoder_inputs, \
            batch_decoder_inputs, batch_weights, sequence_lengths, \
            sequence_lengths


