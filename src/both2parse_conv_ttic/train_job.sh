#!/bin/bash

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/tf_r12/bin/activate
source /home-nfs/ttran/environ

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_conv
TRAIN_DIR=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/both2parse_conv
MODEL_DIR=$TRAIN_DIR/models

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py -data_dir $DATA_DIR -tb_dir $MODEL_DIR >> $MODEL_DIR/output.20170307.txt
#    --eval_dev 


