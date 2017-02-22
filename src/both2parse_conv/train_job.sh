#!/bin/bash

#cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune
#WARM_PATH=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/init_models/t2p_tuned.pickle

#LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" 
python train_many2one.py \
    --warm_path=None \
    --max_epochs=200 >> output.20170221.txt
#    --eval_dev 


