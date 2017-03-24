#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py -data_dir $DATA_DIR -tb_dir $MODEL_DIR -text_num_layers 3 -speech_num_layers 3 -parse_num_layers 3 >> $MODEL_DIR/output.set02.3layers.txt
#    --eval_dev 


