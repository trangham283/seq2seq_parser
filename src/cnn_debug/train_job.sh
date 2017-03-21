#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

# this is an old version, therefore -use_speech was by default
LD_PRELOAD="/homes/ttmt001/transitory/gperftools-2.5/lib/libtcmalloc.so" python train_model.py -run_id 0 -num_filters 5 -filter_sizes 10-50 -data_dir $DATA_DIR -tb_dir $MODEL_DIR >> $MODEL_DIR/output.cnn.small.txt
#    --eval_dev 


