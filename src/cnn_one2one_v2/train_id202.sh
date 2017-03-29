#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

cd /share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/cnn_one2one_v2

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
MODEL_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/models


LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -run_id 202 -use_speech -text_num_layers 3 -parse_num_layers 3 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 64 -filter_sizes 10-25-50 -max_epochs 100 >> $MODEL_DIR/output.id202.txt

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -dump_vars -run_id 92 -use_speech -text_num_layers 3 -parse_num_layers 3 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -num_filters 256 -max_epochs 75 

