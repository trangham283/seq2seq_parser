#!/bin/bash

#DATA_DIR=/scratch/ttran/swbd_data 
DATA_DIR=/tmp/ttran/swbd_data
TCMALLOC_PATH=/home-nfs/ttran/sw/opt/lib/

if [ ! -e ${DATA_DIR} ]                                                         
then  
    mkdir -p ${DATA_DIR}
    cp /share/data/speech/Data/ttran/for_batch_jobs/swbd_data/* ${DATA_DIR} 
fi

source /home-nfs/ttran/transitory/speech-nlp/venv_projects/venv1/bin/activate
cd /home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/

#TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/debug"
#TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/debug_big"
TRAIN_DIR="/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/debug_small_attn_vec"

# using default parameters
#python debug_many2one.py --data_dir ${DATA_DIR}  --train_dir ${TRAIN_DIR} 
#python debug_normal.py --data_dir ${DATA_DIR}  --train_dir ${TRAIN_DIR} 

LD_PRELOAD="/home-nfs/ttran/sw/opt/lib/libtcmalloc.so" python debug_many2one.py \
    --data_dir ${DATA_DIR}  --train_dir ${TRAIN_DIR} \
    --batch_size=4 \
    --num_layers=3 \
    --hidden_size=256 \
    --embedding_size=512 \
    --max_steps=60

