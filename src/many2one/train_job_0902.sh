#!/bin/bash

#DATA_DIR=/scratch/ttran/swbd_speech 
DATA_DIR=/tmp/ttran/swbd_speech

if [ ! -e ${DATA_DIR} ]                                                         
then  
    mkdir -p ${DATA_DIR}
    cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_speech/*.pickle ${DATA_DIR} 
fi

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/ven4/bin/activate
source /home-nfs/ttran/environ
#source /home-nfs/ttran/transitory/speech-nlp/venv_projects/venv1/bin/activate
cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/

TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-many2one-0903"

# using default parameters
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python train_many2one.py \
    --data_dir ${DATA_DIR} \
    --train_dir ${TRAIN_DIR} \
    --steps_per_checkpoint=250 \
    --batch_size=128 \
    --max_steps=80000 >> sbatch_output_many2one_0903.txt
deactivate

    
