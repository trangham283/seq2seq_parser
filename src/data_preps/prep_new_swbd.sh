#!/bin/bash

DATA_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_speech
OUTPUT_DIR=/share/data/speech/Data/ttran/for_batch_jobs/swbd_new

python prep_encoder_data.py --data_dir=$DATA_DIR --output_dir=$OUTPUT_DIR

