#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

cd /share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/cnn_one2one_opt

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
MODEL_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/models
BEST_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/best_models

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_model.py -run_id 1300 -use_speech -num_filters 64 -filter_sizes 10-25-50 -use_conv -data_dir $DATA_DIR -tb_dir $MODEL_DIR -bm_dir $BEST_DIR -max_epochs 100 >> $MODEL_DIR/output.id1300.txt


