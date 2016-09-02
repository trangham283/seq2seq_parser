#!/bin/bash

DATA_DIR=/tmp/ttran/swbd_data

if [ ! -e ${DATA_DIR} ]
then
    mkdir -p ${DATA_DIR}
    cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_data/* ${DATA_DIR}
fi

#source /home-nfs/ttran/transitory/speech-nlp/venv_projects/venv1/bin/activate
source /home-nfs/ttran/transitory/speech-nlp/venv_projects/ven4/bin/activate
source /home-nfs/ttran/environ

cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/text_2_parse
TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-seqlen-0902"
#TRAIN_DIR='/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/text_2_parse'

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_swbd_only.py \
    --data_dir ${DATA_DIR} \
    --train_dir ${TRAIN_DIR} \
    --attention \
    --steps_per_checkpoint=500 \
    --batch_size=128 \
    --max_steps=100000 >> output.swbd_seqlen_0902.txt
