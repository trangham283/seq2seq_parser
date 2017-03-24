#!/bin/bash

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_new

#source /home-nfs/ttran/transitory/speech-nlp/venv_projects/venv1/bin/activate
source /home-nfs/ttran/transitory/speech-nlp/venv_projects/ven4/bin/activate
source /home-nfs/ttran/environ

cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune
TRAIN_DIR=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_tune/models/
WARM_PATH=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/init_models/t2p_tuned.pickle

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_many2one.py \
    --data_dir=${DATA_DIR} \
    --train_base_dir=${TRAIN_DIR} \
    --warm_start_path=${WARM_PATH} \
    --lstm \
    --embedding_size=512 \
    --text_num_layers=2 \
    --text_hidden_size=500 \
    --speech_num_layers=1 \
    --speech_hidden_size=500 \
    --parse_num_layers=2 \
    --parse_hidden_size=500 \
    --output_keep_prob=0.7 \
    --steps_per_checkpoint=250 \
    --batch_size=64 \
    --run_id=2 \
    --max_epochs=200  >> output.many2one_runid_2.txt
#    --eval_dev 


