#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
#python train_many2one_zero.py -data_dir $DATA_DIR -tb_dir $MODEL_DIR -max_epochs 10 -run_id 98 >> $MODEL_DIR/output.speech0_98.txt
python train_many2one_zero.py -data_dir $DATA_DIR -tb_dir $MODEL_DIR -max_epochs 10 -run_id 98 -eval


