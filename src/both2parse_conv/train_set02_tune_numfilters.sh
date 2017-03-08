#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 15 -run_id 2 >> $MODEL_DIR/output.set02.numfilters_15.txt
#    --eval_dev 


