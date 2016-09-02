#!/bin/bash

#DATA_DIR=/scratch/ttran/swbd_data 
DATA_DIR=/tmp/ttran/swbd_data

if [ ! -e ${DATA_DIR} ]                                                         
then  
    mkdir -p ${DATA_DIR}
    cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_data/* ${DATA_DIR} 
fi

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/ven4/bin/activate
source /home-nfs/ttran/environ
cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/

TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-0801"

# using default parameters
LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python parse_nn_swbd.py \
    --data_dir ${DATA_DIR} \
    --train_dir ${TRAIN_DIR} \
    --attention \
    --steps_per_checkpoint=250 \
    --max_steps=100000 >> sbatch_output_0801.txt 
deactivate

    
