#!/bin/bash

TARGET_DIR=/scratch/ttran/Datasets/swtotal_data
RAW_DIR=/scratch/ttran/Datasets/parses_swbd
BK_DIR=/scratch/ttran/Datasets/bktotal_data

# process sentences (input)
python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab90000.sents \
  --tokenizer_name=None \
  --max_vocabulary_size=90000 \
  --data_path=${RAW_DIR}/train.sents \
  --target_path=${TARGET_DIR}/swbd.train.data.ids90000.sents

python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab90000.sents \
  --tokenizer_name=None \
  --max_vocabulary_size=90000 \
  --data_path=${RAW_DIR}/dev.sents \
  --target_path=${TARGET_DIR}/swbd.dev.data.ids90000.sents

python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab90000.sents \
  --tokenizer_name=None \
  --max_vocabulary_size=90000 \
  --data_path=${RAW_DIR}/dev2.sents \
  --target_path=${TARGET_DIR}/swbd.dev2.data.ids90000.sents

# reprocess previous parser data
python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab90000.sents \
  --tokenizer_name=None \
  --max_vocabulary_size=90000 \
  --data_path=${BK_DIR}/train.data.sents \
  --target_path=${TARGET_DIR}/bktotal.train.data.ids90000.sents

#######################################
# process parses (output)
python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab128.parse \
  --tokenizer_name=None \
  --max_vocabulary_size=128 \
  --data_path=${RAW_DIR}/train.parse \
  --target_path=${TARGET_DIR}/swbd.train.data.ids128.parse

python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab128.parse \
  --tokenizer_name=None \
  --max_vocabulary_size=128 \
  --data_path=${RAW_DIR}/dev.parse \
  --target_path=${TARGET_DIR}/swbd.dev.data.ids128.parse

python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab128.parse \
  --tokenizer_name=None \
  --max_vocabulary_size=128 \
  --data_path=${RAW_DIR}/dev2.parse \
  --target_path=${TARGET_DIR}/swbd.dev2.data.ids128.parse


# reprocess previous parser data
python do_data_utils.py \
  --vocabulary_path=${TARGET_DIR}/vocab128.parse \
  --tokenizer_name=None \
  --max_vocabulary_size=128 \
  --data_path=${BK_DIR}/train.data.parse \
  --target_path=${TARGET_DIR}/bktotal.train.data.ids128.parse


