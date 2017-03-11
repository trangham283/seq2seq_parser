#!/bin/bash

MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models
TRAIN_DIR=$MODEL_DIR/text_only_prevbest
#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_model.py --num_layers=3 --batch_size=128 --embedding_size=512 --train_dir=$TRAIN_DIR >> $MODEL_DIR/output.t2p.prevbest.txt
