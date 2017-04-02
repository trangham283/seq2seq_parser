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

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_many2one.py \
    --data_dir=${DATA_DIR} \
    --train_base_dir=${TRAIN_DIR} \
    --lstm \
    --speech_bucket_scale=10 \
    --embedding_size=512 \
    --speech_num_layers=2 \
    --output_keep_prob=0.8 \
    --run_id=99 \
    --steps_per_checkpoint=250 >> output.jobid.99.txt
#    --eval_dev 


