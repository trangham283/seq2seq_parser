#!/bin/bash

DATA_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/data
MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

python train_model.py -run_id 50 -use_conv -use_speech -tb_dir $MODEL_DIR -esize 32 -text_hsize 32 -parse_hsize 32 -conv_filter 5 -conv_channel 2 -num_filters 3 -filter_sizes 5-10 -max_epochs 2 -num_check 10

#    --eval_dev 


