#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

cd /share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/cnn_one2one_pros

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
MODEL_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/models
BEST_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/best_models

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -run_id 10 -use_speech -use_conv -conv_filter 10 -conv_channel 1 -text_hsize 64 -parse_hsize 64 -psize 4 -esize 64 -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -num_filters 4 -filter_sizes 10-50 -max_epochs 2 >> $MODEL_DIR/output.id10.txt


