#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

python train_model.py -tb_dir $MODEL_DIR -esize 64 -text_hsize 64 -parse_hsize 64 -conv_filter 10 -conv_channel 2 -num_filters 2 -max_epochs 2 -num_check 10 -use_conv -run_id 47 



#    --eval_dev 


