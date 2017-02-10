#!/bin/bash

step=160001

eval_path='/share/data/speech/Data/ttran/parser_misc/SParseval/src/sparseval'
prm_file='/share/data/speech/Data/ttran/parser_misc/SParseval/SPEECHPAR.prm'

model_dir=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-lstm-warm-0907

gold_file=${model_dir}/gold-step${step}.txt
br_file=${model_dir}/decoded-br-step${step}.txt
mx_file=${model_dir}/decoded-mx-step${step}.txt

$eval_path -p $prm_file $gold_file $mx_file 
