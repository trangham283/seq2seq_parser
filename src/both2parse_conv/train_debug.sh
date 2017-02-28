#!/bin/bash

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py --batch_size=8 --max_epochs=12 --embedding_size=50 --attention_vector_size=16 --text_hidden_size=16  --speech_hidden_size=16  --parse_hidden_size=16 --num_filters=2 --run_id=1 --eval_dev

