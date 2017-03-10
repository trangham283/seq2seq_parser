#!/bin/bash

MODEL_DIR=/g/ssli/transitory/ttmt001/seq2seq_parser/models

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_model.py --num_layers=3 --batch_size=64 >> $MODEL_DIR/output.t2p.layers.txt
