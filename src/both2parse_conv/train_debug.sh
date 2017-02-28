#!/bin/bash

#cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune
#WARM_PATH=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/init_models/t2p_tuned.pickle

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py \
    --batch_size=8
    --max_epochs=10 \
    --embedding_size=50 \
    --attention_vector_size=16 \
    --text_hidden_size=16 \
    --speech_hidden_size=16 \
    --parse_hidden_size=16 \
    --num_filters=2

#>> output.20170227.txt
#    --eval_dev 


