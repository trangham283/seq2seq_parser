#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_model.py -run_id 3 -use_conv -text_num_layers 3 -parse_num_layers 3 -esize 512 -conv_filter 40 -conv_channel 40 -data_dir $DATA_DIR -tb_dir $MODEL_DIR >> $MODEL_DIR/output.prevbest_conv_no_speech.txt
#    --eval_dev 


