#!/bin/bash

#cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune
#WARM_PATH=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/init_models/t2p_tuned.pickle

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py \
    --max_epochs=10 \
    --embedding_size=100 \
    --attention_vector_size=32 \
    --text_hidden_size=32 \
    --speech_hidden_size=32 \
    --parse_hidden_size=32 \
    --num_filters=2

#>> output.20170227.txt
#    --eval_dev 


