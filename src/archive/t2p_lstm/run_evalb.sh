#!/bin/bash

step=9849

evalb_path='/share/data/speech/Data/ttran/parser_misc/EVALB/evalb'
prm_file='/share/data/speech/Data/ttran/parser_misc/EVALB/seq2seq.prm'

model_dir=/home-nfs/ttran/transitory/speech-nlp/venv_projects/seq2seq_parser/tmp_results/model-swbd-lstm-0905/

#gold_file=${model_dir}/gold-step${step}.txt
#br_file=${model_dir}/decoded-br-step${step}.txt
#mx_file=${model_dir}/decoded-mx-step${step}.txt

gold_file=debug.gold.txt
br_file=debug.decoded.br.txt
mx_file=debug.decoded.mx.txt

$evalb_path -p $prm_file $gold_file $br_file > out_br.txt
$evalb_path -p $prm_file $gold_file $mx_file > out_mx.txt
