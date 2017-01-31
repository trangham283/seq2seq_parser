#!/bin/bash
DATA_DIR=/tmp/ttran/Datasets/audio_swbd

if [ ! -e ${DATA_DIR} ]
then
    mkdir -p ${DATA_DIR}
    cp -r /share/data/speech/Data/ttran/for_batch_jobs/sph ${DATA_DIR}
fi

/home-nfs/ttran/sw/kaldi/egs/swbd/s5c/comp_fbank_energy.sh
cp -r /tmp/ttran/Datasets/audio_swbd/fbank /share/data/speech/Data/ttran/for_batch_jobs/

