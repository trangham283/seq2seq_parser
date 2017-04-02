#!/bin/bash

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_tune

#if [ ! -e ${DATA_DIR} ]
#then
#    mkdir -p ${DATA_DIR}
#    cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_new/* ${DATA_DIR}
#fi
#cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_new/vocab* ${DATA_DIR}

#source /home-nfs/ttran/transitory/speech-nlp/venv_projects/venv1/bin/activate
source /home-nfs/ttran/transitory/speech-nlp/venv_projects/ven4/bin/activate
source /home-nfs/ttran/environ

cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_extra
TRAIN_DIR=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/many2one_extra/models/
WARM_PATH=/share/data/speech/Data/ttran/speech-nlp/venv_projects/seq2seq_parser/many2one_extra/t2p_tuned.pickle

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_many2one.py \
    --data_dir=${DATA_DIR} \
    --train_base_dir=${TRAIN_DIR} \
    --warm_start_path=${WARM_PATH} \
    --lstm \
    --embedding_size=512 \
    --speech_num_layers=1 \
    --output_keep_prob=0.7 \
    --run_id=4 \
    --steps_per_checkpoint=250 >> output.jobid.04.txt
#    --eval_dev 


